#include "pocl-vortex.h"
#include "pocl_builtin_kernels.h"
#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"

#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#include "pocl_context.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "common_driver.h"
#include "pocl_llvm.h"

#include "vortex_utils.h"
#include "kernel_args.h"
// vortex2.h is the canonical async runtime API; the Vortex device driver
// uses it exclusively (no legacy vortex.h entry points).
#include <vortex2.h>

typedef struct {
  vx_device_h vx_device;
  // Queue for the vortex2 async path: buffer DMA + kernel launch.
  vx_queue_h  vx_queue;

  /* List of commands ready to be executed */
  _cl_command_node *ready_list;

  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;

  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  pocl_lock_t compile_lock;

  /* Async command worker. Commands must execute off the caller's stack:
   * pocl_create_migration_commands() submits migrations while POCL holds a
   * buffer lock, and the inline execution path re-entered that lock via
   * pocl_free_event_memobjs() -> self-deadlock (seen on opencl/copybuf). */
  pocl_thread_t worker_thread;
  pocl_cond_t   cq_cond;       /* signaled when ready_list gains work */
  pocl_cond_t   idle_cond;     /* broadcast when the device goes idle */
  int           command_running;
  int           exit_requested;

  int is_64bit;

  /* VX_CAPS_ISA_FLAGS as reported by the device (MISA std bits | arch | Vortex
   * extension bits). Queried once at init, validated against what this driver
   * compiles and emits (vortex_check_isa), and consulted afterwards instead of
   * assuming any unit is present. */
  uint64_t isa_flags;

  /* Fixed-function TEX unit present (VX_ISA_EXT_TEX). When set, image objects
   * are allocated as physical/pinned (the TEX unit reads through tcache,
   * bypassing the per-core MMU) so FF-eligible reads can be routed through the
   * hardware sampler (Tier A). When clear, every image samples in software. */
  int has_tex;

  /* Device extension / OpenCL-C feature strings reported through CL_DEVICE_EXTENSIONS
   * and CL_DEVICE_OPENCL_C_FEATURES. Derived from isa_flags at init (not from the
   * build-time VORTEX_DEVICE_EXTENSIONS constant, which cannot know what the device
   * implements), owned by the driver, freed at uninit. */
  char *extensions;
  char *features;

  size_t ctx_refcount;

  /* Monotonic per-device program index. Each built program links its device
   * code at a distinct base (see compile_vortex_program) so multiple programs
   * in one context (e.g. hybridsort) don't overlap. Guarded by compile_lock. */
  unsigned module_slot;
} vortex_device_data_t;

static void *pocl_vortex_driver_thread (void *arg);

// Block until a vortex2 enqueue event completes, then release it.
// POCL's buffer ops (read/write/copy) are synchronous from the caller's
// view, so each enqueue is followed immediately by this wait. Returns 0
// on success.
static int vx_sync_event(vx_event_h ev) {
  if (ev == NULL)
    return 0;
  int r = vx_event_wait_value(ev, 1, VX_TIMEOUT_INFINITE);
  vx_event_release(ev);
  return r;
}

typedef struct {
  /* The compiled .vxbin as a vortex2 module. Each OpenCL kernel is its own
   * named entry point inside it, resolved per-kernel by pocl_vortex_create
   * _kernel via vx_module_get_kernel. */
  vx_module_h vx_module;
} vortex_program_data_t;

typedef struct {
  size_t refcount;
  /* This kernel's entry point in the program module. */
  vx_kernel_h vx_kernel;
} vortex_kernel_data_t;

typedef struct {
  vx_device_h vx_device;
  vx_buffer_h vx_buffer;
  uint64_t buf_address;
  /* For image mem objects: a device buffer holding the (lifetime-constant)
   * dev_image_t descriptor, built + uploaded once at allocation and referenced
   * by every launch — so binding an image adds no per-launch host<->device
   * round trip. NULL / 0 for plain buffers. */
  vx_buffer_h img_desc_buffer;
  uint64_t    img_desc_address;
} vortex_buffer_data_t;

/* ---- Fixed-function TEX (Tier-A image sampling) ABI ------------------------
 * These are the device-side contract (VX_types.toml [dcr_tex]/[tex_const] in the
 * Vortex RTL/SimX tree). Hardcoded rather than pulled from VX_types.h because the
 * PoCL runtime is built against one Vortex install but the DCR writes are decoded
 * by whichever libvortex/SimX the process loads; these values are stable ABI.
 */
#define VX_DCR_TEX_STAGE   0x040u  /* stage-select (latches the target bank)     */
#define VX_DCR_TEX_ADDR    0x041u  /* texture base >> 6 (64-byte blocks)         */
#define VX_DCR_TEX_LOGDIM  0x042u  /* (log2h << 16) | log2w                      */
#define VX_DCR_TEX_FORMAT  0x043u  /* VX_TEX_FORMAT_*                            */
#define VX_DCR_TEX_FILTER  0x044u  /* POINT=0 / BILINEAR=1 (| MIP_LINEAR=2)      */
#define VX_DCR_TEX_WRAP    0x045u  /* (wrap_v << 16) | wrap_u                    */
#define VX_DCR_TEX_MIPOFF0 0x046u  /* per-LOD byte offset; LOD0 = 0 (base)       */

#define VX_TEX_FMT_A8R8G8B8 0u     /* only FF format the OpenCL mapping uses now  */
#define VX_TEX_STAGE_COUNT  2u

#define VX_TEXF_POINT     0u
#define VX_TEXF_BILINEAR  1u
#define VX_TEXW_CLAMP     0u       /* CLK_ADDRESS_CLAMP_TO_EDGE / NONE           */
#define VX_TEXW_REPEAT    1u
#define VX_TEXW_MIRROR    2u

/* Bit layout of the dev_sampler_t scalar (pocl_fill_dev_sampler_t): bit0 =
 * normalized-coords, bits[3:1] = address mode (CLK_ADDRESS_*), bits[5:4] =
 * filter (CLK_FILTER_*). These mirror the CLK_* values the kernel decodes. */
#define CLK_ADDR_NONE            0x00u
#define CLK_ADDR_CLAMP_TO_EDGE   0x02u
#define CLK_ADDR_CLAMP           0x04u   /* clamp-to-border: not FF, -> software */
#define CLK_ADDR_REPEAT          0x06u
#define CLK_ADDR_MIRRORED_REPEAT 0x08u
#define CLK_FILT_NEAREST         0x10u
#define CLK_FILT_LINEAR          0x20u

/* Map an OpenCL image format to an FF-decodable VX_TEX_FORMAT. Returns 1 and
 * sets *vx_fmt when Tier-A can decode it, else 0 (software sampling). Phase 1
 * covers the hot 8-bit-UNORM BGRA/RGBA case; the FF unit always emits A8R8G8B8
 * and the kernel swizzles per channel order. Other formats stay software. */
static int vx_tex_map_format(const cl_image_format* fmt, uint32_t* vx_fmt) {
  if (fmt->image_channel_data_type != CL_UNORM_INT8)
    return 0;
  if (fmt->image_channel_order != CL_RGBA && fmt->image_channel_order != CL_BGRA)
    return 0;
  *vx_fmt = VX_TEX_FMT_A8R8G8B8;
  return 1;
}

/* Translate the dev_sampler_t bits into FF filter/wrap. Returns 1 when the
 * sampler is FF-representable (nearest/linear × clamp-to-edge/repeat/mirror),
 * else 0 -> software (e.g. clamp-to-border, which the FF unit cannot do). */
static int vx_tex_map_sampler(uint32_t smp, uint32_t* filter, uint32_t* wrap) {
  switch (smp & 0x30u) {
  case CLK_FILT_NEAREST: *filter = VX_TEXF_POINT;    break;
  case CLK_FILT_LINEAR:  *filter = VX_TEXF_BILINEAR; break;
  default: return 0;
  }
  switch (smp & 0x0eu) {
  case CLK_ADDR_NONE:
  case CLK_ADDR_CLAMP_TO_EDGE:   *wrap = VX_TEXW_CLAMP;  break;
  case CLK_ADDR_REPEAT:          *wrap = VX_TEXW_REPEAT; break;
  case CLK_ADDR_MIRRORED_REPEAT: *wrap = VX_TEXW_MIRROR; break;
  default: return 0; /* CLK_ADDRESS_CLAMP (border) is not FF-representable */
  }
  return 1;
}

/* Integer log2 for POT dims; returns -1 if not a power of two. */
static int vx_ilog2_pot(uint32_t v) {
  if (v == 0 || (v & (v - 1)) != 0) return -1;
  int l = 0; while (v > 1) { v >>= 1; ++l; }
  return l;
}

/* ---- Batched CP submission (Phase 3) --------------------------------------
 * The prism runtime exposes vx_enqueue_commands(): submit an ordered list of CP
 * commands (DCR-register writes + a kernel launch) as ONE CP ring batch — a
 * single doorbell and one completion — the compute analog of the graphics
 * CMD_DRAW batch. Binding TEX stages + dispatching then costs one host<->device
 * round trip instead of one per DCR plus one for the launch. SDK headers that
 * predate this entry point declare none of it, so mirror its (stable) ABI here;
 * it resolves against the loaded libvortex like the rest of the vortex2 ABI.
 * VORTEX_HAS_ENQUEUE_COMMANDS is set by CMake after probing the SDK header --
 * the declaration is an enum + struct + prototype, none of which a preprocessor
 * guard in this file can test for. */
#ifndef VORTEX_HAS_ENQUEUE_COMMANDS
typedef enum {
  VX_COMMAND_LAUNCH    = 0,
  VX_COMMAND_DCR_WRITE = 1,
} vx_command_type_e;
typedef struct {
  vx_command_type_e type;
  union {
    const vx_launch_info_t* launch;
    struct { uint32_t addr; uint32_t value; } dcr;
  } data;
} vx_command_t;
extern vx_result_t vx_enqueue_commands(vx_queue_h q, const vx_command_t* commands,
                                       uint32_t count, uint32_t n_wait_events,
                                       const vx_event_h* wait_events,
                                       vx_event_h* out_event);
#endif

/* Max TEX-bind DCRs foldable into one launch batch: 7 per stage x 2 stages. */
#define VX_TEX_MAX_BIND_DCRS (VX_TEX_STAGE_COUNT * 7u)

static cl_bool vortex_available = CL_TRUE;

static const char *vortex_native_device_aux_funcs[] = {NULL};

/* ---- Device capability gate ------------------------------------------------
 * This driver is not ISA-agnostic: it compiles kernels for a fixed RISC-V target,
 * links a kernel library that emits raw AMO instructions (lib/kernel/vortex/
 * atomics.c) and hard floating point, and optionally routes image sampling through
 * the fixed-function TEX unit. None of that may be assumed -- Vortex is configurable
 * and ships with 'A', 'C', 'D', LMEM and every fixed-function unit individually
 * switchable. So VX_CAPS_ISA_FLAGS is queried once at init and every dependent
 * decision is keyed off it, in two tiers:
 *
 *   fatal    -- the device cannot run *any* kernel this driver would emit: an XLEN
 *               mismatch, a missing base extension (I/M/F, plus D under the RV64
 *               lp64d ABI), or a pervasive extension that the configured build flags
 *               tell clang to spray through all code ('C', Zicond). Device init
 *               fails with a diagnostic rather than letting kernels trap at dispatch.
 *   gated    -- the device runs kernels, just not those using the missing feature:
 *               'A' (atomics), LMEM (__local), TEX (FF image sampling). The feature
 *               is switched off and un-advertised, so an OpenCL app querying
 *               CL_DEVICE_EXTENSIONS / _OPENCL_C_FEATURES / _LOCAL_MEM_SIZE sees what
 *               the hardware actually implements instead of a hardcoded claim.
 *
 * The non-fatal tier matters: the stock Vortex config has 'A' disabled (atomics are
 * an opt-in -DVX_CFG_EXT_A_ENABLE build), and OpenCL programs that never touch an
 * atomic run fine on it.
 */

/* MISA 'M' (bit 12). vortex2.h declares the std bits this driver needs except M. */
#ifndef VX_ISA_STD_M
#define VX_ISA_STD_M (1ull << 12)
#endif

/* Device XLEN encoded in VX_CAPS_ISA_FLAGS[31:30] (MISA MXL); VX_ISA_ARCH()
 * expands it to 32 / 64. */
#define VORTEX_ISA_XLEN(flags) ((unsigned)VX_ISA_ARCH (flags))

typedef struct {
  uint64_t bit;
  const char *name;
  /* Non-NULL: the device is refused without it. Text says what breaks. */
  const char *required_by;
} vortex_isa_feature_t;

/* Every ISA bit the driver requires, keys a feature off, or reports. The optional
 * fixed-function units (RASTER/OM/TCU/DXA/RTU) are decoded for the capability log
 * only; the OpenCL path issues none of their instructions. */
static const vortex_isa_feature_t vortex_isa_features[] = {
  { VX_ISA_STD_I, "I", "base integer ISA" },
  { VX_ISA_STD_M, "M", "kernel code emits mul/div throughout (index math)" },
  { VX_ISA_STD_F, "F", "single-precision float; the ilp32f/lp64d ABI passes "
                       "floats in FP registers" },
  /* D is fatal on RV64 only (the lp64d ABI passes doubles in FPRs); on RV32 the ABI
   * is ilp32f and 'D' only gates double-precision codegen. Special-cased below. */
  { VX_ISA_STD_D, "D", NULL },
  /* Gated, not fatal: see vortex_build_extensions / dev->local_mem_size / has_tex. */
  { VX_ISA_STD_A, "A", NULL },
  { VX_ISA_STD_C, "C", NULL },
  { VX_ISA_EXT_LMEM, "LMEM", NULL },
  { VX_ISA_EXT_TEX, "TEX", NULL },
#ifdef VX_ISA_EXT_RASTER
  { VX_ISA_EXT_RASTER, "RASTER", NULL },
#endif
#ifdef VX_ISA_EXT_OM
  { VX_ISA_EXT_OM, "OM", NULL },
#endif
#ifdef VX_ISA_EXT_TCU
  { VX_ISA_EXT_TCU, "TCU", NULL },
#endif
#ifdef VX_ISA_EXT_DXA
  { VX_ISA_EXT_DXA, "DXA", NULL },
#endif
#ifdef VX_ISA_EXT_RTU
  { VX_ISA_EXT_RTU, "RTU", NULL },
#endif
};

/* Fatal tier: does the device implement what every kernel this driver emits needs?
 * Returns CL_SUCCESS, or CL_INVALID_DEVICE naming the missing capability. Reported on
 * stderr as well as through POCL_MSG_ERR: rejecting the device must be visible without
 * POCL_DEBUG, or it just looks like "no OpenCL device". */
static cl_int
vortex_isa_check (uint64_t isa_flags, int is_64bit)
{
  unsigned want_xlen = is_64bit ? 64u : 32u;
  unsigned have_xlen = VORTEX_ISA_XLEN (isa_flags);
  size_t i;
  int missing = 0;

  if (have_xlen != want_xlen)
    {
      POCL_MSG_ERR ("Vortex: device is RV%u, POCL_VORTEX_XLEN selects RV%u.\n",
                    have_xlen, want_xlen);
      fprintf (stderr,
               "pocl-vortex: device is RV%u but POCL_VORTEX_XLEN=%u -- ISA mismatch, "
               "device disabled.\n", have_xlen, want_xlen);
      return CL_INVALID_DEVICE;
    }

  for (i = 0; i < sizeof (vortex_isa_features) / sizeof (vortex_isa_features[0]); ++i)
    {
      const vortex_isa_feature_t *f = &vortex_isa_features[i];
      const char *why = f->required_by;

      if (why == NULL && is_64bit && f->bit == VX_ISA_STD_D)
        why = "the lp64d ABI passes doubles in FP registers";
      if (why == NULL || (isa_flags & f->bit))
        continue;

      POCL_MSG_ERR ("Vortex: device lacks required ISA extension '%s' (%s).\n",
                    f->name, why);
      fprintf (stderr,
               "pocl-vortex: device lacks required ISA extension '%s' (%s) -- "
               "kernels would trap; device disabled.\n", f->name, why);
      missing = 1;
    }

  return missing ? CL_INVALID_DEVICE : CL_SUCCESS;
}

/* CL_DEVICE_EXTENSIONS, derived from the device ISA. VORTEX_DEVICE_EXTENSIONS is a
 * build-time constant that cannot know what the device implements, so the
 * device-dependent claims in it are filtered here:
 *   cl_khr_int64                 -- 64-bit scalars need an RV64 device
 *   cl_khr_int64_*_atomics       -- 64-bit AMOs need 'A' on an RV64 device
 *   cl_khr_*_int32_*_atomics     -- 32-bit AMOs need 'A'
 * The rest of the list is device-independent (cl_khr_byte_addressable_store) or
 * gated by the build config (cl_khr_il_program, cl_ext_buffer_device_address).
 * Caller owns the returned string. */
static char *
vortex_build_extensions (uint64_t isa_flags, int is_64bit)
{
  /* Room for the build-time list plus every runtime-gated token. */
  size_t cap = strlen (VORTEX_DEVICE_EXTENSIONS) + 128;
  char *exts = (char *)calloc (cap, 1);
  int has_atomics = (isa_flags & VX_ISA_STD_A) != 0;
  const char *tok;
  char *save = NULL;
  char *build_exts;

  if (exts == NULL)
    return NULL;

  build_exts = strdup (VORTEX_DEVICE_EXTENSIONS);
  if (build_exts == NULL)
    {
      free (exts);
      return NULL;
    }

  for (tok = strtok_r (build_exts, " ", &save); tok != NULL;
       tok = strtok_r (NULL, " ", &save))
    {
      /* Drop the device-dependent claims the build-time list makes unconditionally. */
      if (strcmp (tok, "cl_khr_int64") == 0 && !is_64bit)
        continue;
      if (strstr (tok, "atomics") != NULL && !has_atomics)
        continue;
      /* 64-bit AMOs additionally need an RV64 device. NOTE: the Vortex kernel library
       * implements the 32-bit AMO builtins only (lib/kernel/vortex/atomics.c), so even
       * on a device that clears this gate a kernel calling atom_add(long*) fails to
       * link -- loudly, at build time, rather than silently mis-executing. */
      if ((strcmp (tok, "cl_khr_int64_base_atomics") == 0
           || strcmp (tok, "cl_khr_int64_extended_atomics") == 0)
          && !is_64bit)
        continue;
      if (exts[0] != '\0')
        strncat (exts, " ", cap - strlen (exts) - 1);
      strncat (exts, tok, cap - strlen (exts) - 1);
    }

  /* cl_khr_fp64 is not advertised even when the device implements 'D': the Vortex
   * kernel library provides no double-precision builtins, so it would be a claim the
   * toolchain cannot honor. The 'D' bit only gates codegen (pocl_vortex_init_build). */

  free (build_exts);
  return exts;
}

/* CL_DEVICE_OPENCL_C_FEATURES, derived from the same ISA flags: an app that queries
 * features rather than extensions must see the same truth. Only features the device
 * and the Vortex kernel library can both back are listed -- notably no
 * __opencl_c_fp64 (no double builtins) and no __opencl_c_atomic_* without 'A'.
 * Caller owns the returned string. */
static char *
vortex_build_features (uint64_t isa_flags, int is_64bit, int image_support)
{
  char buf[512];
  buf[0] = '\0';

  if (image_support)
    strncat (buf, " __opencl_c_images", sizeof (buf) - strlen (buf) - 1);
  if (is_64bit)
    strncat (buf, " __opencl_c_int64", sizeof (buf) - strlen (buf) - 1);
  if (isa_flags & VX_ISA_STD_A)
    strncat (buf, " __opencl_c_atomic_order_relaxed __opencl_c_atomic_scope_device",
             sizeof (buf) - strlen (buf) - 1);

  return strdup (buf[0] == ' ' ? buf + 1 : buf);
}

/* Log the decoded ISA once, so a capability-driven behavior change (software vs FF
 * image sampling, a dropped extension) is explainable from the device's own report. */
static void
vortex_log_isa (uint64_t isa_flags)
{
  char buf[256];
  size_t i;
  int n;

  n = snprintf (buf, sizeof (buf), "RV%u", VORTEX_ISA_XLEN (isa_flags));
  for (i = 0; i < sizeof (vortex_isa_features) / sizeof (vortex_isa_features[0]); ++i)
    {
      const vortex_isa_feature_t *f = &vortex_isa_features[i];
      if ((isa_flags & f->bit) && n > 0 && (size_t)n < sizeof (buf) - 1)
        n += snprintf (buf + n, sizeof (buf) - (size_t)n, " %s", f->name);
    }
  POCL_MSG_PRINT_INFO ("Vortex device ISA: %s (flags 0x%llx)\n", buf,
                       (unsigned long long)isa_flags);
}

/* Release everything acquired so far on a failed init. */
static cl_int
vortex_init_fail (vx_device_h vx_device, vortex_device_data_t *dd, cl_int err)
{
  if (vx_device != NULL)
    vx_device_release (vx_device);
  if (dd != NULL)
    {
      POCL_MEM_FREE (dd->extensions);
      POCL_MEM_FREE (dd->features);
      free (dd);
    }
  return err;
}

/* The final device compile (compile_vortex_program) drives clang with the user's
 * POCL_VORTEX_CFLAGS, whose -march decides which instructions actually land in the
 * .vxbin. Check that march against the device ISA at init, so an ISA the hardware
 * cannot execute is rejected up front rather than trapping at the first dispatch.
 * Recognized: the std letters (imafdc) plus the Vortex/RISC-V target features this
 * flow uses (+zicond). Anything unrecognized is left alone -- this is a guard, not
 * an march parser. */
static cl_int
vortex_check_build_flags (uint64_t isa_flags, int is_64bit)
{
  const char *cflags = pocl_get_string_option ("POCL_VORTEX_CFLAGS", "");
  const char *march = strstr (cflags, "-march=rv");
  const char *p;
  unsigned want_xlen;

  /* Not set yet: compile_vortex_program errors out on its own at build time. */
  if (march == NULL)
    return CL_SUCCESS;

  march += strlen ("-march=rv");
  want_xlen = (strncmp (march, "64", 2) == 0) ? 64u : 32u;
  if (want_xlen != (is_64bit ? 64u : 32u))
    {
      fprintf (stderr,
               "pocl-vortex: POCL_VORTEX_CFLAGS builds RV%u but the device is RV%u "
               "-- device disabled.\n", want_xlen, is_64bit ? 64u : 32u);
      return CL_INVALID_DEVICE;
    }

  for (p = march + 2; *p && *p != ' ' && *p != '_'; ++p)
    {
      uint64_t bit = 0;
      /* 'c' is pervasive: with it in -march, clang compresses instructions all over
       * the kernel, so a device without 'C' traps immediately. 'a' and 'd' are only
       * emitted where a kernel actually uses an atomic or a double -- a mismatch
       * there is a per-kernel build failure, not a reason to refuse the device
       * (the stock Vortex config ships without 'A'). */
      int pervasive = 0;
      switch (*p)
        {
        case 'i': bit = VX_ISA_STD_I; pervasive = 1; break;
        case 'm': bit = VX_ISA_STD_M; pervasive = 1; break;
        case 'f': bit = VX_ISA_STD_F; pervasive = 1; break;
        case 'c': bit = VX_ISA_STD_C; pervasive = 1; break;
        case 'a': bit = VX_ISA_STD_A; break;
        case 'd': bit = VX_ISA_STD_D; break;
        default: continue; /* 'g', vendor suffixes: not our business */
        }
      if (isa_flags & bit)
        continue;
      if (pervasive)
        {
          fprintf (stderr,
                   "pocl-vortex: POCL_VORTEX_CFLAGS requests -march=...%c... but the "
                   "device does not implement RISC-V '%c' -- device disabled.\n",
                   *p, *p);
          return CL_INVALID_DEVICE;
        }
      POCL_MSG_WARN ("Vortex: -march requests '%c' which the device lacks; kernels "
                     "using it will fail to build.\n", *p);
    }

  /* Zicond is pervasive as well: clang selects czero.eqz/nez for any select. */
  if (strstr (cflags, "+zicond") != NULL && !(isa_flags & VX_ISA_EXT_ZICOND))
    {
      fprintf (stderr,
               "pocl-vortex: POCL_VORTEX_CFLAGS enables +zicond but the device does "
               "not implement Zicond -- device disabled.\n");
      return CL_INVALID_DEVICE;
    }

  return CL_SUCCESS;
}

/* Target features for the IR/kernel-library build stage. Derived from the device ISA
 * rather than hardcoded: a device without 'A' must not have clang lower atomics to
 * AMOs, and one without 'D' must not get native double-precision instructions. */
char* pocl_vortex_init_build(void *data) {
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  uint64_t isa_flags = (dd != NULL) ? dd->isa_flags : 0;
  char flags[256];

  /* M and F are mandatory (vortex_isa_check), so they are always present here. */
  snprintf (flags, sizeof (flags),
            "-target-feature +m -target-feature +f%s%s",
            (isa_flags & VX_ISA_STD_A) ? " -target-feature +a" : "",
            (isa_flags & VX_ISA_STD_D) ? " -target-feature +d" : "");
  return strdup (flags);
}

void pocl_vortex_init_device_ops(struct pocl_device_ops *ops) {

  ops->device_name = "vortex";
  ops->build_hash = pocl_vortex_build_hash;
  ops->probe = pocl_vortex_probe;
  ops->uninit = pocl_vortex_uninit;
  ops->init = pocl_vortex_init;

  ops->init_context = pocl_vortex_init_context;
  ops->free_context = pocl_vortex_free_context;

  ops->run = pocl_vortex_run;
  ops->run_native = NULL;

  ops->alloc_mem_obj = pocl_vortex_alloc_mem_obj;
  ops->free = pocl_vortex_free;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_driver_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  /* No build_poclbinary: the Vortex device keeps its program as LLVM
   * bitcode (program->binaries) and recompiles to a Vortex ELF in
   * pocl_vortex_run; it does not produce poclbinary WG-function
   * containers. pocl_driver_build_poclbinary needs ops->compile_kernel,
   * which the Vortex device deliberately does not implement -- calling
   * it NULL-derefs (seen via chipStar's clGetProgramInfo cache path).
   * Leaving this NULL makes clGetProgramInfo(CL_PROGRAM_BINARIES)
   * serialize the bitcode program directly. */
  ops->build_poclbinary = NULL;
  ops->build_builtin = pocl_driver_build_opencl_builtins;
  ops->init_build = pocl_vortex_init_build;

  ops->post_build_program = pocl_vortex_post_build_program;
  ops->free_program = pocl_vortex_free_program;

  ops->create_kernel = pocl_vortex_create_kernel;
  ops->free_kernel = pocl_vortex_free_kernel;

  ops->submit = pocl_vortex_submit;
  ops->join = pocl_vortex_join;
  ops->flush = pocl_vortex_flush;
  ops->notify = pocl_vortex_notify;
  ops->broadcast = pocl_broadcast;

  ops->read = pocl_vortex_read;
  ops->read_rect = pocl_vortex_read_rect;
  ops->write = pocl_vortex_write;
  ops->write_rect = pocl_vortex_write_rect;
  ops->copy = pocl_vortex_copy;
  ops->copy_rect = pocl_vortex_copy_rect;
  ops->memfill = pocl_vortex_memfill;

  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;

  /* clEnqueueMapBuffer/UnmapMemObject: Vortex has separate device memory, so
   * map/unmap must DMA device<->host (the generic pocl_driver_map_mem does a
   * host memcpy from the device address and segfaults). Without any handler,
   * pocl_exec_command calls a NULL ops->map_mem and segfaults (e.g. hotspot). */
  ops->map_mem = pocl_vortex_map_mem;
  ops->unmap_mem = pocl_vortex_unmap_mem;

  /* Image transfers. Vortex device memory is not host-addressable, so the
   * generic basic/driver rect image ops (which memcpy through the device
   * address as a host pointer) cannot be reused. These wrappers reuse the
   * Vortex rect DMA path (pocl_vortex_read/write/copy_rect) instead, and
   * fill_image stages the pattern through a host buffer. Sampling itself runs
   * in software via the read_image.cl builtins compiled into the Vortex kernel
   * library; see the pocl_image_support proposal. */
  ops->read_image_rect  = pocl_vortex_read_image_rect;
  ops->write_image_rect = pocl_vortex_write_image_rect;
  ops->copy_image_rect  = pocl_vortex_copy_image_rect;
  ops->map_image        = pocl_vortex_map_image;
  ops->unmap_image      = pocl_vortex_unmap_image;
  ops->fill_image       = pocl_vortex_fill_image;
}

char * pocl_vortex_build_hash (cl_device_id dev)
{
  char *res = (char *)calloc(1000, sizeof(char));
  if (dev->address_bits == 64) {
    snprintf(res, 1000, "vortex-riscv64-unknown-unknown-elf");
  } else {
    snprintf(res, 1000, "vortex-riscv32-unknown-unknown-elf");
  }
  return res;
}

unsigned int pocl_vortex_probe(struct pocl_device_ops *ops)
{
  return (0 == strcmp(ops->device_name, "vortex"));
}

cl_int
pocl_vortex_init (unsigned j, cl_device_id dev, const char* parameters)
{
  int vx_err;
  vortex_device_data_t *dd;

  const char* sz_xlen = pocl_get_string_option("POCL_VORTEX_XLEN", "32");

  int is_64bit = (strcmp(sz_xlen, "64") == 0);

  assert (dev->data == NULL);

  dd = (vortex_device_data_t *)calloc(1, sizeof(vortex_device_data_t));
  if (dd == NULL){
    return CL_OUT_OF_HOST_MEMORY;
  }

  /* The device must be opened and its ISA validated before the device infos are
   * populated: the reported extension string (and the fp configs POCL derives from
   * it) depend on what the hardware actually implements. */
  vx_device_h vx_device;
  vx_err = vx_device_open(0, &vx_device);
  if (vx_err != 0) {
    return vortex_init_fail(NULL, dd, CL_DEVICE_NOT_FOUND);
  }

  uint64_t isa_flags = 0;
  vx_err = vx_device_query(vx_device, VX_CAPS_ISA_FLAGS, &isa_flags);
  if (vx_err != 0) {
    POCL_MSG_ERR("Vortex: VX_CAPS_ISA_FLAGS query failed (%d).\n", vx_err);
    fprintf(stderr, "pocl-vortex: cannot read the device ISA capabilities -- "
                    "device disabled.\n");
    return vortex_init_fail(vx_device, dd, CL_DEVICE_NOT_FOUND);
  }
  dd->isa_flags = isa_flags;

  cl_int caps_err = vortex_isa_check(isa_flags, is_64bit);
  if (caps_err == CL_SUCCESS)
    caps_err = vortex_check_build_flags(isa_flags, is_64bit);
  if (caps_err != CL_SUCCESS) {
    return vortex_init_fail(vx_device, dd, caps_err);
  }

  vortex_log_isa(isa_flags);

  if (!(isa_flags & VX_ISA_STD_A)) {
    /* Not fatal (the stock Vortex config has no 'A'), but the OpenCL 1.2 core atomic
     * builtins cannot be served: they are inline amo*.w (lib/kernel/vortex/atomics.c).
     * Un-advertised below; a kernel that calls one fails to build. */
    POCL_MSG_WARN("Vortex: device has no 'A' extension; OpenCL atomics are "
                  "unavailable (rebuild the device with -DVX_CFG_EXT_A_ENABLE).\n");
  }

  dd->extensions = vortex_build_extensions(isa_flags, is_64bit);
  dd->features = vortex_build_features(isa_flags, is_64bit, /*image_support=*/1);
  if (dd->extensions == NULL || dd->features == NULL) {
    return vortex_init_fail(vx_device, dd, CL_OUT_OF_HOST_MEMORY);
  }

  pocl_init_default_device_infos(dev, dd->extensions);
  dev->features = dd->features;

  SETUP_DEVICE_CL_VERSION (dev, VORTEX_DEVICE_CL_VERSION_MAJOR,
                           VORTEX_DEVICE_CL_VERSION_MINOR);

  dev->vendor = "Vortex Group";
  dev->long_name = "Vortex OpenGPU";
  dev->short_name = "Vortex";
  dev->vendor_id = 0;
  dev->type = CL_DEVICE_TYPE_GPU;

  dev->spmd = CL_TRUE;
  dev->run_workgroup_pass = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;
  //dev->global_as_id = VX_ADDR_SPACE_GLOBAL;
  //dev->local_as_id = VX_ADDR_SPACE_LOCAL;439
  //dev->constant_as_id = VX_ADDR_SPACE_CONSTANT;
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;
  dev->device_alloca_locals = CL_FALSE;
  dev->device_side_printf = 0;
  dev->has_64bit_long = is_64bit;

  /* Phase 1 (pocl_vortex_v3_proposal.md): opt the Vortex device into the
   * SPIR-V code path. POCL's central pipeline parses SPIR-V into LLVM IR
   * (via lib/CL/devices/spirv_parser.{cc,hh}) before calling the
   * device-specific post_build_program hook below, so this opt-in is
   * sufficient -- the existing compile_vortex_program flow continues to
   * accept program->llvm_irs[device_i] regardless of whether the source
   * was OpenCL-C or SPIR-V. Required prerequisite for chipStar
   * (chipstar_on_vortex_proposal.md), since chipStar issues
   * clCreateProgramWithIL with a SPIR-V binary. */
#ifdef ENABLE_SPIRV
  dev->supported_spir_v_versions = "SPIR-V_1.2";
  /* Whitelist the SPIR-V extensions emitted by SPIRV-LLVM-Translator 20
   * for chipStar's HIP -> SPIR-V output. Without this, POCL passes
   * --spirv-ext=-all to llvm-spirv and the back-translate fails with
   * "Invalid SPIR-V module: input SPIR-V module uses extension X which
   * were disabled by --spirv-ext option". Observed exts on the chipStar
   * vecadd path:
   *   SPV_KHR_untyped_pointers       -- opaque-pointer LLVM IR
   *   SPV_INTEL_fp_fast_math_mode    -- fast-math attrs on FP ops
   * Adding the common SPIR-V hygiene exts as well; they are no-ops if
   * the input doesn't use them. */
  dev->supported_spirv_extensions = "+SPV_KHR_no_integer_wrap_decoration"
                                    ",+SPV_KHR_untyped_pointers"
                                    ",+SPV_INTEL_fp_fast_math_mode"
                                    ",+SPV_EXT_shader_atomic_float_add"
                                    ",+SPV_INTEL_memory_access_aliasing"
                                    ",+SPV_INTEL_inline_assembly";
#else
  dev->supported_spir_v_versions = "";
#endif

  dev->llvm_cpu = NULL;
  dev->address_bits = is_64bit ? 64 : 32;
  dev->llvm_target_triplet = is_64bit ? "riscv64-unknown-unknown-elf" : "riscv32-unknown-unknown-elf";
  dev->llvm_abi = is_64bit ? "lp64d" : "ilp32f";
  dev->llvm_cpu = is_64bit ? "generic-rv64" : "generic-rv32";
  dev->kernellib_name = is_64bit ? "kernel-riscv64" : "kernel-riscv32";
  dev->kernellib_fallback_name = NULL;
  dev->kernellib_subdir = "vortex";
  dev->device_aux_functions = vortex_native_device_aux_funcs;

  /* Image support is served in software: the read/write_image + get_image_*
   * builtins are compiled into the Vortex kernel library, and the six image
   * transfer ops are wired in pocl_vortex_init_device_ops. The image caps
   * (max_*_image_args, image2d/3d limits, supported_image_formats table) are
   * already populated by pocl_init_default_device_infos() above; enabling the
   * flag exposes them. */
  dev->image_support = CL_TRUE;

  uint64_t num_cores;
  vx_err = vx_device_query(vx_device, VX_CAPS_NUM_CORES, &num_cores);
  if (vx_err != 0) {
    return vortex_init_fail(vx_device, dd, CL_DEVICE_NOT_FOUND);
  }

  uint64_t global_mem_size;
  vx_err = vx_device_query(vx_device, VX_CAPS_GLOBAL_MEM_SIZE, &global_mem_size);
  if (vx_err != 0) {
    return vortex_init_fail(vx_device, dd, CL_DEVICE_NOT_FOUND);
  }

  uint64_t local_mem_size;
  vx_err = vx_device_query(vx_device, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size);
  if (vx_err != 0) {
    return vortex_init_fail(vx_device, dd, CL_DEVICE_NOT_FOUND);
  }

  uint64_t num_warps;
  vx_err = vx_device_query(vx_device, VX_CAPS_NUM_WARPS, &num_warps);
  if (vx_err != 0) {
    return vortex_init_fail(vx_device, dd, CL_DEVICE_NOT_FOUND);
  }

  uint64_t num_threads;
  vx_err = vx_device_query(vx_device, VX_CAPS_NUM_THREADS, &num_threads);
  if (vx_err != 0) {
    return vortex_init_fail(vx_device, dd, CL_DEVICE_NOT_FOUND);
  }

  /* Async DMA queue for the vortex2 buffer path. */
  vx_queue_h vx_queue;
  vx_queue_info_t qinfo;
  memset(&qinfo, 0, sizeof(qinfo));
  qinfo.struct_size = sizeof(qinfo);
  qinfo.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_err = vx_queue_create(vx_device, &qinfo, &vx_queue);
  if (vx_err != 0) {
    return vortex_init_fail(vx_device, dd, CL_DEVICE_NOT_FOUND);
  }

  uint64_t max_work_group_size = num_warps * num_threads;

  dev->global_mem_size = global_mem_size;
  dev->max_mem_alloc_size = global_mem_size;
  /* VX_CAPS_LOCAL_MEM_SIZE is decoded from the config word whether or not the
   * scratchpad is built, so report a size only when the device actually has one.
   * Kernels with __local args then fail at enqueue instead of addressing nothing. */
  if (isa_flags & VX_ISA_EXT_LMEM) {
    dev->local_mem_size = local_mem_size;
    dev->local_mem_type = CL_LOCAL;
  } else {
    POCL_MSG_WARN("Vortex: device has no local memory (LMEM disabled); kernels "
                  "using __local will not run.\n");
    dev->local_mem_size = 0;
    dev->local_mem_type = CL_GLOBAL;
  }
  dev->max_work_group_size    = max_work_group_size;
  dev->max_work_item_sizes[0] = max_work_group_size;
  dev->max_work_item_sizes[1] = max_work_group_size;
  dev->max_work_item_sizes[2] = max_work_group_size;
  dev->max_compute_units = num_cores;

  dd->vx_device = vx_device;
  dd->vx_queue  = vx_queue;

  dd->ctx_refcount = 0;

  dd->is_64bit = is_64bit;

  /* Fixed-function TEX unit (Tier-A image sampling). Optional: when absent, every
   * image samples in software, so this gates the feature rather than the device. */
  dd->has_tex = (isa_flags & VX_ISA_EXT_TEX) ? 1 : 0;

  POCL_INIT_LOCK(dd->compile_lock);
  POCL_INIT_LOCK(dd->cq_lock);
  POCL_INIT_COND(dd->cq_cond);
  POCL_INIT_COND(dd->idle_cond);
  dd->command_running = 0;
  dd->exit_requested = 0;
  POCL_CREATE_THREAD(dd->worker_thread, &pocl_vortex_driver_thread, dd);

  dev->data = dd;
  dev->available = &vortex_available;

  return CL_SUCCESS;
}

cl_int pocl_vortex_uninit (unsigned j, cl_device_id dev) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  if (NULL == dd)
    return CL_SUCCESS;

  POCL_LOCK (dd->cq_lock);
  dd->exit_requested = 1;
  POCL_SIGNAL_COND (dd->cq_cond);
  POCL_UNLOCK (dd->cq_lock);
  POCL_JOIN_THREAD (dd->worker_thread);

  /* Perf counters dumped once at device teardown. The legacy launch path
   * dumped them per-run when freeing the previous kernel buffer; with the
   * module loaded once at build time there is no longer a per-run hook. */
  vx_device_dump_perf(dd->vx_device, stdout);
  vx_queue_release(dd->vx_queue);
  vx_device_release(dd->vx_device);

  POCL_DESTROY_COND (dd->cq_cond);
  POCL_DESTROY_COND (dd->idle_cond);
  POCL_DESTROY_LOCK (dd->compile_lock);
  POCL_DESTROY_LOCK (dd->cq_lock);
  /* dev->extensions/features point into dd (built from the device ISA at init). */
  dev->extensions = NULL;
  dev->features = NULL;
  POCL_MEM_FREE(dd->extensions);
  POCL_MEM_FREE(dd->features);
  POCL_MEM_FREE(dd);
  dev->data = NULL;
  return CL_SUCCESS;
}

int pocl_vortex_init_context (cl_device_id dev, cl_context context) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  if (NULL == dd) {
    /* A previous free_context tore the device down when its last context
     * closed (see pocl_vortex_free_context). Multi-program apps that release
     * one context and create another (e.g. b+tree's two kernel wrappers,
     * hybridsort) would then find dev->data == NULL and crash in the next
     * build. Re-open the device lazily so each fresh context gets a live one. */
    cl_int r = pocl_vortex_init(0, dev, NULL);
    if (r != CL_SUCCESS)
      return r;
    dd = (vortex_device_data_t *)dev->data;
    if (NULL == dd)
      return CL_SUCCESS;
  }

  dd->ctx_refcount++;

  return CL_SUCCESS;
}

int pocl_vortex_free_context (cl_device_id dev, cl_context context) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  if (NULL == dd)
    return CL_SUCCESS;

  if (--dd->ctx_refcount == 0) {
    pocl_vortex_uninit(0, dev);
  }

  return CL_SUCCESS;
}

int pocl_vortex_post_build_program (cl_program program, cl_uint device_i) {
  int result;
  cl_device_id dev = program->devices[device_i];
  vortex_device_data_t *ddata = (vortex_device_data_t *)dev->data;
  vortex_program_data_t *pdata = NULL;

  POCL_LOCK (ddata->compile_lock);

  do {
    result = pocl_llvm_run_passes_on_program (program, device_i);
    if (result != 0)
      break;

    /* Capability gate (see vortex_isa_check): a device without 'A' cannot execute the
     * AMOs that the OpenCL atomic builtins compile to, and neither -march nor the
     * assembler rejects them (the kernel library reaches them through inline asm).
     * Fail the build with the reason in the build log rather than letting the device
     * abort mid-kernel on an illegal instruction. */
    if (!(ddata->isa_flags & VX_ISA_STD_A)
        && vortex_module_uses_atomics (program->llvm_irs[device_i])) {
      const char *msg = "error: the program uses OpenCL atomics, but the Vortex "
                        "device does not implement the RISC-V 'A' extension "
                        "(rebuild the device with -DVX_CFG_EXT_A_ENABLE).\n";
      pocl_append_to_buildlog (program, device_i, strdup (msg), strlen (msg));
      POCL_MSG_ERR ("Vortex: %s", msg);
      result = CL_BUILD_PROGRAM_FAILURE;
      break;
    }

    pdata = (vortex_program_data_t *)calloc (1, sizeof (vortex_program_data_t));

    char sz_program_bc[POCL_MAX_PATHNAME_LENGTH];
    char sz_program_vxbin[POCL_MAX_PATHNAME_LENGTH];

    pocl_cache_program_bc_path(sz_program_bc, program, device_i);
    remove_extension(sz_program_bc);

    strcpy(sz_program_vxbin, sz_program_bc);
    strncat(sz_program_vxbin, ".vxbin", POCL_MAX_PATHNAME_LENGTH - 1);

    result = compile_vortex_program(sz_program_vxbin,
                                    program->llvm_irs[device_i],
                                    ddata->module_slot);
    if (result != 0)
      break;
    /* Consume this slot only on a successful build so a failed compile
     * doesn't leak a code region. */
    ddata->module_slot++;

    /* Load the freshly-compiled .vxbin as a vortex2 module up front. Each
     * kernel is its own named entry point; pocl_vortex_create_kernel
     * resolves them individually. */
    result = vx_module_load_file(ddata->vx_device, sz_program_vxbin,
                                 &pdata->vx_module);
    if (result != 0)
      break;

  } while (0);

  program->data[device_i] = pdata;

  POCL_UNLOCK (ddata->compile_lock);

  return result;
}

int pocl_vortex_free_program (cl_device_id dev, cl_program program,
                              unsigned device_i) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  vortex_program_data_t *pdata = (vortex_program_data_t *)program->data[device_i];
  if (pdata == NULL)
    return CL_SUCCESS;

  pocl_driver_free_program (dev, program, device_i);

  /* Per-kernel vx_kernel handles are released by pocl_vortex_free_kernel. */
  if (pdata->vx_module != NULL)
    vx_module_release (pdata->vx_module);

  POCL_MEM_FREE (pdata);
  program->data[device_i] = NULL;

  return CL_SUCCESS;
}

int pocl_vortex_create_kernel (cl_device_id dev, cl_program program,
                               cl_kernel kernel, unsigned device_i) {
  int result = CL_SUCCESS;
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  vortex_kernel_data_t *kdata  = (vortex_kernel_data_t *)meta->data[device_i];
  if (kdata != NULL) {
    ++kdata->refcount;
    return CL_SUCCESS;
  }

  do {
    vortex_program_data_t *pdata = (vortex_program_data_t *)program->data[device_i];
    assert(pdata != NULL);

    /* Resolve this kernel's named entry point in the program module. */
    vx_kernel_h vx_kernel = NULL;
    if (vx_module_get_kernel(pdata->vx_module, kernel->name, &vx_kernel) != 0) {
      POCL_MSG_ERR("vortex: kernel '%s' not found in module\n", kernel->name);
      result = CL_INVALID_KERNEL_NAME;
      break;
    }
    kdata = (vortex_kernel_data_t *)calloc (1, sizeof (vortex_kernel_data_t));
    kdata->vx_kernel = vx_kernel;
    ++kdata->refcount;

  } while (0);

  meta->data[device_i] = kdata;

  return result;
}

int pocl_vortex_free_kernel (cl_device_id dev, cl_program program,
                             cl_kernel kernel, unsigned device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  vortex_kernel_data_t *kdata = (vortex_kernel_data_t *)meta->data[device_i];
  if (kdata == NULL)
    return CL_SUCCESS;

  --kdata->refcount;
  if (kdata->refcount == 0) {
    if (kdata->vx_kernel != NULL)
      vx_kernel_release (kdata->vx_kernel);
    POCL_MEM_FREE (kdata);
    meta->data[device_i] = NULL;
  }

  return CL_SUCCESS;
}

/* Tier-A TEX binder (see call site in pocl_vortex_run). For each FF-eligible
 * image arg it (a) collects the stage-programming DCR writes into dcr_addr/
 * dcr_val — the caller folds them into the launch's CP batch (one doorbell) —
 * and (b) patches the image's cached descriptor {_tex_stage,_tex_sampler},
 * resetting the others to unbound. Returns the number of DCR writes collected
 * (0 without TEX, so the launch stays a plain vx_enqueue_launch). The descriptor
 * patch is a small synchronous write that must land before the launch reads it;
 * folding it into the batch too awaits the runtime's batch-safe mem-write path. */
static uint32_t vx_tex_bind_images(vortex_device_data_t* dd, _cl_command_node* cmd,
                                   pocl_kernel_metadata_t* meta, uint32_t ptr_size,
                                   uint32_t* dcr_addr, uint32_t* dcr_val) {
  if (!dd->has_tex)
    return 0;
  uint32_t nd = 0;
  #define VX_TEX_EMIT(A, V) do { dcr_addr[nd] = (A); dcr_val[nd] = (V); ++nd; } while (0)

  unsigned mem_id = cmd->device->global_mem_id;

  /* One FF-representable sampler drives the binding (the common image kernel has
   * exactly one). Images sampled with any other sampler fail the kernel-side
   * match and sample in software — safe, never wrong. */
  int have_smp = 0;
  uint32_t smp_bits = 0, smp_filter = 0, smp_wrap = 0;
  for (int i = 0; i < meta->num_args; ++i) {
    if (meta->arg_info[i].type != POCL_ARG_TYPE_SAMPLER)
      continue;
    dev_sampler_t ds = 0;
    pocl_fill_dev_sampler_t(&ds, &cmd->command.run.arguments[i]);
    uint32_t bits = (uint32_t)ds, f, w;
    if (vx_tex_map_sampler(bits, &f, &w)) {
      have_smp = 1; smp_bits = bits; smp_filter = f; smp_wrap = w;
      break;
    }
  }

  const uint32_t patch_off = ptr_size + 12u * 4u; /* {_tex_stage,_tex_sampler} */
  uint32_t next_stage = 0;
  for (int i = 0; i < meta->num_args; ++i) {
    if (meta->arg_info[i].type != POCL_ARG_TYPE_IMAGE)
      continue;
    cl_mem m = *(cl_mem*)(cmd->command.run.arguments[i].value);
    vortex_buffer_data_t* bd =
        (vortex_buffer_data_t*)m->device_ptrs[mem_id].extra_ptr;

    int32_t patch[2] = { -1, 0 }; /* unbound by default */

    uint32_t vx_fmt = 0;
    cl_image_format fmt = { m->image_channel_order, m->image_channel_data_type };
    int lw = vx_ilog2_pot((uint32_t)m->image_width);
    int lh = vx_ilog2_pot((uint32_t)m->image_height);
    if (have_smp && bd && next_stage < VX_TEX_STAGE_COUNT
        && m->type == CL_MEM_OBJECT_IMAGE2D
        && lw >= 0 && lh >= 0
        /* FF addressing assumes tightly-packed rows. image_elem_size is
         * bytes-per-channel, so the pixel stride is channels * elem_size. */
        && m->image_row_pitch
             == (size_t)m->image_width * m->image_channels * m->image_elem_size
        && vx_tex_map_format(&fmt, &vx_fmt)
        && (m->flags & CL_MEM_WRITE_ONLY) == 0
        && (bd->buf_address & 63u) == 0) {
      uint32_t stage = next_stage++;
      VX_TEX_EMIT(VX_DCR_TEX_STAGE,  stage);
      VX_TEX_EMIT(VX_DCR_TEX_LOGDIM, ((uint32_t)lh << 16) | (uint32_t)lw);
      VX_TEX_EMIT(VX_DCR_TEX_FORMAT, vx_fmt);
      VX_TEX_EMIT(VX_DCR_TEX_FILTER, smp_filter);
      VX_TEX_EMIT(VX_DCR_TEX_WRAP,   (smp_wrap << 16) | smp_wrap);
      VX_TEX_EMIT(VX_DCR_TEX_ADDR,   (uint32_t)(bd->buf_address >> 6));
      VX_TEX_EMIT(VX_DCR_TEX_MIPOFF0, 0);
      patch[0] = (int32_t)stage;
      patch[1] = (int32_t)smp_bits;
      if (getenv("POCL_VORTEX_TEX_DEBUG"))
        fprintf(stderr, "[vortex-tex] bound image arg %d -> stage %u "
                "(logdim=%dx%d fmt=%u filt=%u wrap=%u smp=0x%x addr=0x%lx)\n",
                i, stage, lw, lh, vx_fmt, smp_filter, smp_wrap, smp_bits,
                (unsigned long)bd->buf_address);
    } else if (getenv("POCL_VORTEX_TEX_DEBUG")) {
      fprintf(stderr, "[vortex-tex] image arg %d NOT FF-bound "
              "(have_smp=%d stage_avail=%d 2d=%d pot=%d,%d pitch=%zu/%zu "
              "fmt_ok=%d wo=%d aligned=%d)\n",
              i, have_smp, next_stage < VX_TEX_STAGE_COUNT,
              m->type == CL_MEM_OBJECT_IMAGE2D, lw, lh,
              (size_t)m->image_row_pitch,
              (size_t)m->image_width * m->image_channels * m->image_elem_size,
              vx_tex_map_format(&fmt, &vx_fmt), (m->flags & CL_MEM_WRITE_ONLY) != 0,
              bd ? (int)((bd->buf_address & 63u) == 0) : -1);
    }

    if (bd && bd->img_desc_buffer) {
      vx_event_h ev = NULL;
      if (vx_enqueue_write(dd->vx_queue, bd->img_desc_buffer, patch_off,
                           patch, sizeof(patch), 0, NULL, &ev) == 0)
        vx_sync_event(ev);
    }
  }
  #undef VX_TEX_EMIT
  return nd;
}

void pocl_vortex_run (void *data, _cl_command_node *cmd) {
  vortex_device_data_t *dd;
  struct pocl_argument *al;
  cl_uint device_i = cmd->program_device_i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  vortex_kernel_data_t *kdata = (vortex_kernel_data_t *)meta->data[device_i];
  struct pocl_context *pc = &cmd->command.run.pc;
  int vx_err;

  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < pc->work_dim; ++i) {
    num_groups *= pc->num_groups[i];
    group_size *= pc->local_size[i];
  }
  if (num_groups == 0 || group_size == 0)
    return;

  assert (data != NULL);
  dd = (vortex_device_data_t *)data;

  uint32_t ptr_size = dd->is_64bit ? 8 : 4;

  uint32_t aligned_kernel_args_size = ALIGN_OFFSET(sizeof(kernel_args_t), ptr_size);

  // calculate kernel arguments buffer size
  uint32_t local_mem_size = 0;
  size_t abuf_size = 0;

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      local_mem_size += al->size;
      abuf_size = ALIGN_OFFSET(abuf_size + 4, ptr_size);
    } else
    if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
     || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
     || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
      abuf_size = ALIGN_OFFSET(abuf_size + ptr_size, ptr_size);
    } else {
      // scalar argument
      abuf_size = ALIGN_OFFSET(abuf_size + al->size, ptr_size);
    }
  }

  // local buffers
  for (int i = 0; i < meta->num_locals; ++i) {
    local_mem_size += meta->local_sizes[i];
    abuf_size = ALIGN_OFFSET(abuf_size + 4, ptr_size);
  }

  // add local size
  if (local_mem_size != 0) {
    abuf_size = ALIGN_OFFSET(abuf_size + 4, ptr_size);
  }

  // allocate arguments host buffer
  size_t kargs_buffer_size = aligned_kernel_args_size + abuf_size;
  uint8_t* const host_kargs_base_ptr = malloc(kargs_buffer_size);
  assert(host_kargs_base_ptr);

  // write context data
  // num_groups/local_size are carried in vx_launch_info_t.grid_dim/block_dim
  // rather than kernel_args_t because the KMU surfaces them through
  // VX_CSR_CTA_{GRID_DIM,BLOCK_DIM}_*. work_dim and global_offset stay in
  // kernel_args_t because the KMU does not expose them.
  {
    kernel_args_t* const kargs = (kernel_args_t*)host_kargs_base_ptr;
    kargs->work_dim = pc->work_dim;
    for (int i = 0; i < 3; ++i) {
      kargs->global_offset[i] = pc->global_offset[i];
    }
  }

  /* Tier-A binding pass: route FF-eligible image reads through the hardware TEX
   * unit. Pick one FF-representable sampler (the common kernel has exactly one),
   * then for every image arg either bind it to a free TEX stage (collecting the
   * stage DCRs and patching its cached descriptor with {stage, sampler}) or
   * reset it to unbound. Resetting every launch prevents a stale stage index —
   * whose DCR may now hold a different texture — from surviving into this one.
   * The collected DCRs are submitted together with the launch as one CP batch
   * (Phase 3, below). */
  uint32_t tex_dcr_addr[VX_TEX_MAX_BIND_DCRS];
  uint32_t tex_dcr_val[VX_TEX_MAX_BIND_DCRS];
  uint32_t tex_n_dcr = vx_tex_bind_images(dd, cmd, meta, ptr_size,
                                          tex_dcr_addr, tex_dcr_val);

  // write arguments

  uint8_t* const host_args_ptr = host_kargs_base_ptr + aligned_kernel_args_size;
  uint32_t host_args_offset = 0;
  uint32_t local_mem_offset = 0;

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      if (local_mem_offset == 0) {
        memcpy(host_args_ptr + host_args_offset, &local_mem_size, 4); // local_size
        host_args_offset = ALIGN_OFFSET(host_args_offset + 4, ptr_size);
      }
      memcpy(host_args_ptr + host_args_offset, &local_mem_offset, 4); // arg offset
      host_args_offset = ALIGN_OFFSET(host_args_offset + 4, ptr_size);
      local_mem_offset += al->size;
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      if (al->value == NULL) {
        memset(host_args_ptr + host_args_offset, 0, ptr_size); // NULL pointer value
        host_args_offset = ALIGN_OFFSET(host_args_offset + ptr_size, ptr_size);
      } else if (al->is_raw_ptr) {
        /* Raw device pointer from clSetKernelArgDevicePointerEXT
         * (cl_ext_buffer_device_address). chipStar's HIP path uses this
         * exclusively: after hipMalloc returns a device address, the
         * kernel arg is set with that raw address rather than a cl_mem*.
         * al->value points to a cl_mem_device_address_EXT (cl_ulong)
         * holding the device-side address — copy it through directly.
         * Pattern matches lib/CL/devices/cuda/pocl-cuda.c is_raw_ptr=1
         * branch. */
        cl_ulong raw_addr;
        memcpy(&raw_addr, al->value, sizeof(cl_ulong));
        raw_addr += al->offset;
        memcpy(host_args_ptr + host_args_offset, &raw_addr, ptr_size);
        host_args_offset = ALIGN_OFFSET(host_args_offset + ptr_size, ptr_size);
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        vortex_buffer_data_t* buf_data = (vortex_buffer_data_t *) m->device_ptrs[cmd->device->global_mem_id].extra_ptr;
        uint64_t dev_mem_addr = buf_data->buf_address + al->offset;
        memcpy(host_args_ptr + host_args_offset, &buf_data->buf_address, ptr_size); // pointer value
        host_args_offset = ALIGN_OFFSET(host_args_offset + ptr_size, ptr_size);
      }
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      /* Pass the device address of the image's cached dev_image_t descriptor,
       * built once at allocation (pocl_vortex_alloc_mem_obj). No per-launch
       * descriptor build/upload/sync — the read_image.cl builtins reinterpret
       * the arg as a global dev_image_t*. */
      cl_mem m = *(cl_mem *)(al->value);
      vortex_buffer_data_t* buf_data =
          (vortex_buffer_data_t *)m->device_ptrs[cmd->device->global_mem_id].extra_ptr;
      memcpy(host_args_ptr + host_args_offset, &buf_data->img_desc_address, ptr_size);
      host_args_offset = ALIGN_OFFSET(host_args_offset + ptr_size, ptr_size);
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      /* The CLK_* sampler bitfield, passed as a pointer-sized scalar; the
       * kernel recovers it via __builtin_astype(sampler, uintptr_t). */
      dev_sampler_t ds = 0;
      pocl_fill_dev_sampler_t(&ds, al);
      uint64_t sv = (uint64_t)ds;
      memcpy(host_args_ptr + host_args_offset, &sv, ptr_size);
      host_args_offset = ALIGN_OFFSET(host_args_offset + ptr_size, ptr_size);
    } else {
      // scalar argument
      memcpy(host_args_ptr + host_args_offset, al->value, al->size); // scalar value
      host_args_offset = ALIGN_OFFSET(host_args_offset + al->size, ptr_size);
    }
  }

  // write local arguments
  for (int i = 0; i < meta->num_locals; ++i) {
    if (local_mem_offset == 0) {
      memcpy(host_args_ptr + host_args_offset, &local_mem_size, 4); // local_size
      host_args_offset = ALIGN_OFFSET(host_args_offset + 4, ptr_size);
    }
    memcpy(host_args_ptr + host_args_offset, &local_mem_offset, 4); // arg offset
    host_args_offset = ALIGN_OFFSET(host_args_offset + 4, ptr_size);
    local_mem_offset += meta->local_sizes[i];
  }

  // launch kernel execution
  /* The vortex2 runtime stages the host args blob into a device-side
   * scratch slot and programs the KMU ARG/PC DCRs itself; the caller no
   * longer allocates, uploads, or frees an args device buffer, nor uploads
   * the kernel per-run (the module is loaded once in post_build_program).
   * The KMU iterates over (block, thread) coordinates in hardware, driven
   * by the grid_dim / block_dim arrays. pc->num_groups and pc->local_size
   * are size_t[3] in POCL's pocl_context struct; copy into uint32_t[3] for
   * vx_launch_info_t. */
  vx_launch_info_t li;
  memset(&li, 0, sizeof(li));
  li.struct_size = sizeof(li);
  li.kernel      = kdata->vx_kernel;
  li.args_host   = host_kargs_base_ptr;
  li.args_size   = kargs_buffer_size;
  li.ndim        = pc->work_dim;
  li.lmem_size   = local_mem_size;
  for (int i = 0; i < 3; ++i) {
    li.grid_dim[i]  = (uint32_t)pc->num_groups[i];
    li.block_dim[i] = (uint32_t)pc->local_size[i];
  }

  vx_event_h ev = NULL;
  if (tex_n_dcr != 0) {
    /* Phase 3: fold the TEX-stage-bind DCR writes and the kernel launch into one
     * CP ring batch — a single doorbell / completion — instead of one enqueue
     * per DCR plus one for the launch. The CP retires them in order (binds
     * before dispatch). Non-TEX launches keep the plain single-launch path. */
    vx_command_t cmds[VX_TEX_MAX_BIND_DCRS + 1];
    uint32_t n = 0;
    for (uint32_t i = 0; i < tex_n_dcr; ++i) {
      cmds[n].type = VX_COMMAND_DCR_WRITE;
      cmds[n].data.dcr.addr = tex_dcr_addr[i];
      cmds[n].data.dcr.value = tex_dcr_val[i];
      ++n;
    }
    cmds[n].type = VX_COMMAND_LAUNCH;
    cmds[n].data.launch = &li;
    ++n;
    vx_err = vx_enqueue_commands(dd->vx_queue, cmds, n, 0, NULL, &ev);
  } else {
    vx_err = vx_enqueue_launch(dd->vx_queue, &li, 0, NULL, &ev);
  }
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // release argument host buffer (held live until the launch completed
  // above, since the runtime stages it at launch time)
  free(host_kargs_base_ptr);
}

cl_int pocl_vortex_alloc_mem_obj(cl_device_id dev, cl_mem mem_obj, void *host_ptr) {
  int vx_err;
  pocl_mem_identifier *p = &mem_obj->device_ptrs[dev->global_mem_id];

  /* let other drivers preallocate */
  if ((mem_obj->flags & CL_MEM_ALLOC_HOST_PTR) && (mem_obj->mem_host_ptr == NULL))
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  p->extra_ptr = NULL;
  p->version = 0;
  p->extra = 0;
  cl_mem_flags flags = mem_obj->flags;

  {
    vortex_device_data_t* dd = (vortex_device_data_t *)dev->data;

    /* CL_MEM_USE_HOST_PTR: Vortex device memory is separate from host memory,
     * so the host allocation cannot be aliased. Emulate it as a device buffer
     * seeded from host_ptr (like COPY_HOST_PTR); POCL retains mem_host_ptr and
     * syncs device->host on read/map. */
    int vx_flags = 0;
    if ((flags & CL_MEM_READ_WRITE) != 0)
      vx_flags = VX_MEM_READ_WRITE;
    if ((flags & CL_MEM_READ_ONLY) != 0)
      vx_flags = VX_MEM_READ;
    if ((flags & CL_MEM_WRITE_ONLY) != 0)
      vx_flags = VX_MEM_WRITE;

    /* Image storage must be physically pinned for the FF TEX unit, which reads
     * through tcache and bypasses the per-core MMU. Pin every image when the
     * device has TEX; the eligibility check in run() decides per launch whether
     * to actually route a given read through the hardware sampler. */
    if (mem_obj->is_image && dd->has_tex)
      vx_flags |= VX_MEM_PHYS;

    vx_buffer_h vx_buffer;
    vx_err = vx_buffer_create(dd->vx_device, mem_obj->size, vx_flags, &vx_buffer);
    if (vx_err != 0) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    uint64_t buf_address;
    vx_err = vx_buffer_address(vx_buffer, &buf_address);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    if (host_ptr && (flags & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR))) {
      vx_event_h ev = NULL;
      vx_err = vx_enqueue_write(dd->vx_queue, vx_buffer, 0, host_ptr,
                                mem_obj->size, 0, NULL, &ev);
      if (vx_err == 0)
        vx_err = vx_sync_event(ev);
      if (vx_err != 0) {
        vx_buffer_release(vx_buffer);
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
    }

    if (flags & CL_MEM_ALLOC_HOST_PTR) {
      /* malloc mem_host_ptr then increase refcount */
      pocl_alloc_or_retain_mem_host_ptr (mem_obj);
    }

    vortex_buffer_data_t* buf_data = (vortex_buffer_data_t *)malloc(sizeof(vortex_buffer_data_t));
    buf_data->vx_device = dd->vx_device;
    buf_data->vx_buffer = vx_buffer;
    buf_data->buf_address = buf_address;
    buf_data->img_desc_buffer = NULL;
    buf_data->img_desc_address = 0;

    /* Image objects: build the dev_image_t descriptor once, here, and upload it
     * to its own small device buffer. The descriptor is constant for the image's
     * lifetime (dims/format/pitch and the pixel-buffer address never change when
     * the contents change), so every launch just references this cached device
     * address — no per-launch descriptor upload or sync. Serialized in
     * device-native layout: a ptr_size pointer (_data) followed by 12 int32
     * fields, because the host dev_image_t uses a 64-bit void* that would not
     * match a 32-bit device. */
    if (mem_obj->is_image) {
      uint32_t ptr_size = dd->is_64bit ? 8 : 4;
      uint8_t desc[8 + 14 * 4];
      uint32_t doff = 0;
      uint64_t data_addr = buf_address;                 // _data = pixel buffer addr
      memcpy(desc + doff, &data_addr, ptr_size);
      doff += ptr_size;
      /* 12 dev_image_t fields + 2 trailing FF-TEX fields (_tex_stage,
       * _tex_sampler). The FF fields default to "unbound" here and are patched
       * per launch by pocl_vortex_run when the image is FF-eligible. */
      const int32_t fields[14] = {
        (int32_t)mem_obj->image_width, (int32_t)mem_obj->image_height,
        (int32_t)mem_obj->image_depth, (int32_t)mem_obj->image_array_size,
        (int32_t)mem_obj->image_row_pitch, (int32_t)mem_obj->image_slice_pitch,
        (int32_t)mem_obj->num_mip_levels, (int32_t)mem_obj->num_samples,
        (int32_t)mem_obj->image_channel_order, (int32_t)mem_obj->image_channel_data_type,
        (int32_t)mem_obj->image_channels, (int32_t)mem_obj->image_elem_size,
        -1 /* _tex_stage: unbound */, 0 /* _tex_sampler */
      };
      memcpy(desc + doff, fields, sizeof(fields));
      doff += (uint32_t)sizeof(fields);

      vx_buffer_h desc_buf;
      if (vx_buffer_create(dd->vx_device, doff, VX_MEM_READ, &desc_buf) != 0) {
        vx_buffer_release(vx_buffer);
        free(buf_data);
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
      uint64_t desc_addr = 0;
      vx_buffer_address(desc_buf, &desc_addr);
      vx_event_h dev_ev = NULL;
      vx_err = vx_enqueue_write(dd->vx_queue, desc_buf, 0, desc, doff, 0, NULL, &dev_ev);
      if (vx_err == 0)
        vx_err = vx_sync_event(dev_ev);
      if (vx_err != 0) {
        vx_buffer_release(desc_buf);
        vx_buffer_release(vx_buffer);
        free(buf_data);
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
      buf_data->img_desc_buffer = desc_buf;
      buf_data->img_desc_address = desc_addr;
    }

    /* Store the real device address in mem_ptr so POCL's CL_MEM_DEVICE_ADDRESS_EXT
     * flow (lib/CL/clCreateBuffer.c around line 258) returns the actual
     * Vortex address to clients like chipStar (which queries it via
     * CL_MEM_DEVICE_ADDRESS_EXT after a CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT
     * allocation). The wrapper struct moves to extra_ptr; all other Vortex
     * sites that need the wrapper now read from extra_ptr. */
    p->mem_ptr = (void *)(uintptr_t)buf_address;
    p->extra_ptr = buf_data;
  }

  return CL_SUCCESS;
}

void pocl_vortex_free(cl_device_id dev, cl_mem mem_obj) {
  pocl_mem_identifier *p = &mem_obj->device_ptrs[dev->global_mem_id];
  cl_mem_flags flags = mem_obj->flags;
  vortex_buffer_data_t* buf_data = (vortex_buffer_data_t*)p->extra_ptr;

  {
    /* USE_HOST_PTR buffers are backed by a device allocation (see alloc);
     * POCL owns the caller's host_ptr, so only release the device buffer. */
    if (flags & CL_MEM_ALLOC_HOST_PTR) {
      pocl_release_mem_host_ptr(mem_obj);
    }
    if (buf_data && buf_data->vx_buffer) {
      vx_buffer_release(buf_data->vx_buffer);
    }
    if (buf_data && buf_data->img_desc_buffer) {
      vx_buffer_release(buf_data->img_desc_buffer);
    }
  }
  if (buf_data) free(buf_data);
  p->mem_ptr = NULL;
  p->extra_ptr = NULL;
  p->version = 0;
}

void pocl_vortex_copy(void *data,
                      pocl_mem_identifier *dst_mem_id,
                      cl_mem dst_buf,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t dst_offset,
                      size_t src_offset,
                      size_t size)
{
  int vx_err;
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *src_buf_data = (vortex_buffer_data_t *)src_mem_id->extra_ptr;
  vortex_buffer_data_t *dst_buf_data = (vortex_buffer_data_t *)dst_mem_id->extra_ptr;
  vx_event_h ev = NULL;
  vx_err = vx_enqueue_copy(dd->vx_queue,
                           dst_buf_data->vx_buffer, dst_offset,
                           src_buf_data->vx_buffer, src_offset,
                           size, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0)
  {
    POCL_ABORT("POCL_VORTEX_COPY\n");
  }
}

void pocl_vortex_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset,
                       size_t size) {
  int vx_err;
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *buf_data = (vortex_buffer_data_t *)dst_mem_id->extra_ptr;
  vx_event_h ev = NULL;
  vx_err = vx_enqueue_write(dd->vx_queue, buf_data->vx_buffer, offset,
                            host_ptr, size, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_WRITE\n");
  }
}

void pocl_vortex_read(void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t offset,
                      size_t size) {
  int vx_err;
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t* buf_data = (vortex_buffer_data_t*)src_mem_id->extra_ptr;
  vx_event_h ev = NULL;
  vx_err = vx_enqueue_read(dd->vx_queue, host_ptr, buf_data->vx_buffer,
                           offset, size, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_READ\n");
  }
}

/* clEnqueueMapBuffer: DMA device->host into the mapping's host staging buffer
 * (allocated by get_mapping_ptr). WRITE_INVALIDATE maps skip the read-in. */
cl_int pocl_vortex_map_mem(void *data, pocl_mem_identifier *src_mem_id,
                           cl_mem src_buf, mem_mapping_t *map) {
  assert(map->host_ptr);
  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *buf_data = (vortex_buffer_data_t *)src_mem_id->extra_ptr;
  vx_event_h ev = NULL;
  int vx_err = vx_enqueue_read(dd->vx_queue, map->host_ptr, buf_data->vx_buffer,
                               map->offset, map->size, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  return (vx_err == 0) ? CL_SUCCESS : CL_MAP_FAILURE;
}

/* clEnqueueUnmapMemObject: DMA host->device unless the mapping was read-only. */
cl_int pocl_vortex_unmap_mem(void *data, pocl_mem_identifier *dst_mem_id,
                             cl_mem dst_buf, mem_mapping_t *map) {
  assert(map->host_ptr);
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *buf_data = (vortex_buffer_data_t *)dst_mem_id->extra_ptr;
  vx_event_h ev = NULL;
  int vx_err = vx_enqueue_write(dd->vx_queue, buf_data->vx_buffer, map->offset,
                                map->host_ptr, map->size, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  return (vx_err == 0) ? CL_SUCCESS : CL_MAP_FAILURE;
}

/* clEnqueue{Read,Write,Copy}BufferRect / clEnqueueFillBuffer. These map
 * 1:1 onto the vortex2 rect/fill enqueues; the runtime owns the strided
 * decomposition (vx_enqueue_*_rect) so POCL no longer slices the rect
 * into N linear transfers itself. For buffer rect ops region[0] and the
 * origin x components are byte counts, matching vx_rect_info_t. */
void pocl_vortex_read_rect(void *data,
                           void *__restrict__ dst_host_ptr,
                           pocl_mem_identifier *src_mem_id,
                           cl_mem src_buf,
                           const size_t *buffer_origin,
                           const size_t *host_origin,
                           const size_t *region,
                           size_t buffer_row_pitch,
                           size_t buffer_slice_pitch,
                           size_t host_row_pitch,
                           size_t host_slice_pitch) {
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *buf_data = (vortex_buffer_data_t *)src_mem_id->extra_ptr;
  vx_rect_info_t rect;
  memset(&rect, 0, sizeof(rect));
  rect.struct_size = sizeof(rect);
  for (int i = 0; i < 3; ++i) {
    rect.buffer_origin[i] = buffer_origin[i];
    rect.host_origin[i]   = host_origin[i];
    rect.region[i]        = region[i];
  }
  rect.buffer_row_pitch   = buffer_row_pitch;
  rect.buffer_slice_pitch = buffer_slice_pitch;
  rect.host_row_pitch     = host_row_pitch;
  rect.host_slice_pitch   = host_slice_pitch;
  vx_event_h ev = NULL;
  int vx_err = vx_enqueue_read_rect(dd->vx_queue, dst_host_ptr,
                                    buf_data->vx_buffer, &rect, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_READ_RECT\n");
  }
}

void pocl_vortex_write_rect(void *data,
                            const void *__restrict__ src_host_ptr,
                            pocl_mem_identifier *dst_mem_id,
                            cl_mem dst_buf,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch) {
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *buf_data = (vortex_buffer_data_t *)dst_mem_id->extra_ptr;
  vx_rect_info_t rect;
  memset(&rect, 0, sizeof(rect));
  rect.struct_size = sizeof(rect);
  for (int i = 0; i < 3; ++i) {
    rect.buffer_origin[i] = buffer_origin[i];
    rect.host_origin[i]   = host_origin[i];
    rect.region[i]        = region[i];
  }
  rect.buffer_row_pitch   = buffer_row_pitch;
  rect.buffer_slice_pitch = buffer_slice_pitch;
  rect.host_row_pitch     = host_row_pitch;
  rect.host_slice_pitch   = host_slice_pitch;
  vx_event_h ev = NULL;
  int vx_err = vx_enqueue_write_rect(dd->vx_queue, buf_data->vx_buffer,
                                     src_host_ptr, &rect, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_WRITE_RECT\n");
  }
}

void pocl_vortex_copy_rect(void *data,
                           pocl_mem_identifier *dst_mem_id,
                           cl_mem dst_buf,
                           pocl_mem_identifier *src_mem_id,
                           cl_mem src_buf,
                           const size_t *dst_origin,
                           const size_t *src_origin,
                           const size_t *region,
                           size_t dst_row_pitch,
                           size_t dst_slice_pitch,
                           size_t src_row_pitch,
                           size_t src_slice_pitch) {
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *dst_data = (vortex_buffer_data_t *)dst_mem_id->extra_ptr;
  vortex_buffer_data_t *src_data = (vortex_buffer_data_t *)src_mem_id->extra_ptr;
  /* vx_rect_info_t carries the destination in buffer_* and the source in
   * host_* for the copy case. */
  vx_rect_info_t rect;
  memset(&rect, 0, sizeof(rect));
  rect.struct_size = sizeof(rect);
  for (int i = 0; i < 3; ++i) {
    rect.buffer_origin[i] = dst_origin[i];
    rect.host_origin[i]   = src_origin[i];
    rect.region[i]        = region[i];
  }
  rect.buffer_row_pitch   = dst_row_pitch;
  rect.buffer_slice_pitch = dst_slice_pitch;
  rect.host_row_pitch     = src_row_pitch;
  rect.host_slice_pitch   = src_slice_pitch;
  vx_event_h ev = NULL;
  int vx_err = vx_enqueue_copy_rect(dd->vx_queue, dst_data->vx_buffer,
                                    src_data->vx_buffer, &rect, 0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_COPY_RECT\n");
  }
}

void pocl_vortex_memfill(void *data,
                         pocl_mem_identifier *dst_mem_id,
                         cl_mem dst_buf,
                         size_t size,
                         size_t offset,
                         const void *__restrict__ pattern,
                         size_t pattern_size) {
  vortex_device_data_t *dd = (vortex_device_data_t *)data;
  vortex_buffer_data_t *buf_data = (vortex_buffer_data_t *)dst_mem_id->extra_ptr;
  vx_event_h ev = NULL;
  int vx_err = vx_enqueue_fill_buffer(dd->vx_queue, buf_data->vx_buffer,
                                      offset, size, pattern, pattern_size,
                                      0, NULL, &ev);
  if (vx_err == 0)
    vx_err = vx_sync_event(ev);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_MEMFILL\n");
  }
}

/* Image transfer ops. Modeled on the basic driver's wrappers, but delegating to
 * the Vortex rect DMA path (pocl_vortex_{read,write,copy}_rect) because Vortex
 * device memory is not host-addressable. The image x-origin and x-region are
 * converted to byte counts (× pixel size), matching what the vortex2 rect
 * enqueues expect (their row 0 / x axis is measured in bytes). */

cl_int pocl_vortex_copy_image_rect(void *data, cl_mem src_image, cl_mem dst_image,
                                   pocl_mem_identifier *src_mem_id,
                                   pocl_mem_identifier *dst_mem_id,
                                   const size_t *src_origin,
                                   const size_t *dst_origin,
                                   const size_t *region) {
  size_t px = src_image->image_elem_size * src_image->image_channels;
  const size_t adj_src_origin[3] = { src_origin[0] * px, src_origin[1], src_origin[2] };
  const size_t adj_dst_origin[3] = { dst_origin[0] * px, dst_origin[1], dst_origin[2] };
  const size_t adj_region[3] = { region[0] * px, region[1], region[2] };
  pocl_vortex_copy_rect(data, dst_mem_id, dst_image, src_mem_id, src_image,
                        adj_dst_origin, adj_src_origin, adj_region,
                        dst_image->image_row_pitch, dst_image->image_slice_pitch,
                        src_image->image_row_pitch, src_image->image_slice_pitch);
  return CL_SUCCESS;
}

cl_int pocl_vortex_write_image_rect(void *data, cl_mem dst_image,
                                    pocl_mem_identifier *dst_mem_id,
                                    const void *__restrict__ src_host_ptr,
                                    pocl_mem_identifier *src_mem_id,
                                    const size_t *origin, const size_t *region,
                                    size_t src_row_pitch, size_t src_slice_pitch,
                                    size_t src_offset) {
  const void *__restrict__ ptr = src_host_ptr ? src_host_ptr : src_mem_id->mem_ptr;
  ptr = (const char *)ptr + src_offset;
  const size_t zero_origin[3] = { 0, 0, 0 };
  size_t px = dst_image->image_elem_size * dst_image->image_channels;
  if (src_row_pitch == 0)
    src_row_pitch = px * region[0];
  if (src_slice_pitch == 0)
    src_slice_pitch = src_row_pitch * region[1];
  const size_t adj_origin[3] = { origin[0] * px, origin[1], origin[2] };
  const size_t adj_region[3] = { region[0] * px, region[1], region[2] };
  pocl_vortex_write_rect(data, ptr, dst_mem_id, dst_image, adj_origin, zero_origin,
                         adj_region, dst_image->image_row_pitch,
                         dst_image->image_slice_pitch, src_row_pitch, src_slice_pitch);
  return CL_SUCCESS;
}

cl_int pocl_vortex_read_image_rect(void *data, cl_mem src_image,
                                   pocl_mem_identifier *src_mem_id,
                                   void *__restrict__ dst_host_ptr,
                                   pocl_mem_identifier *dst_mem_id,
                                   const size_t *origin, const size_t *region,
                                   size_t dst_row_pitch, size_t dst_slice_pitch,
                                   size_t dst_offset) {
  void *__restrict__ ptr = dst_host_ptr ? dst_host_ptr : dst_mem_id->mem_ptr;
  ptr = (char *)ptr + dst_offset;
  const size_t zero_origin[3] = { 0, 0, 0 };
  size_t px = src_image->image_elem_size * src_image->image_channels;
  if (dst_row_pitch == 0)
    dst_row_pitch = px * region[0];
  if (dst_slice_pitch == 0)
    dst_slice_pitch = dst_row_pitch * region[1];
  const size_t adj_origin[3] = { origin[0] * px, origin[1], origin[2] };
  const size_t adj_region[3] = { region[0] * px, region[1], region[2] };
  pocl_vortex_read_rect(data, ptr, src_mem_id, src_image, adj_origin, zero_origin,
                        adj_region, src_image->image_row_pitch,
                        src_image->image_slice_pitch, dst_row_pitch, dst_slice_pitch);
  return CL_SUCCESS;
}

cl_int pocl_vortex_map_image(void *data, pocl_mem_identifier *mem_id,
                             cl_mem src_image, mem_mapping_t *map) {
  assert(map->host_ptr != NULL);
  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;
  if (map->host_ptr != ((char *)mem_id->mem_ptr + map->offset))
    pocl_vortex_read_image_rect(data, src_image, mem_id, map->host_ptr, NULL,
                                map->origin, map->region, map->row_pitch,
                                map->slice_pitch, 0);
  return CL_SUCCESS;
}

cl_int pocl_vortex_unmap_image(void *data, pocl_mem_identifier *mem_id,
                               cl_mem dst_image, mem_mapping_t *map) {
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;
  if (map->host_ptr != ((char *)mem_id->mem_ptr + map->offset))
    pocl_vortex_write_image_rect(data, dst_image, mem_id, map->host_ptr, NULL,
                                 map->origin, map->region, map->row_pitch,
                                 map->slice_pitch, 0);
  return CL_SUCCESS;
}

cl_int pocl_vortex_fill_image(void *data, cl_mem image,
                              pocl_mem_identifier *mem_id, const size_t *origin,
                              const size_t *region, cl_uint4 orig_pixel,
                              pixel_t fill_pixel, size_t pixel_size) {
  /* Replicate the fill pixel across the region in a host staging buffer, then
   * DMA it into the device image via the rect path (device memory can't be
   * memset in place from the host). */
  size_t count = region[0] * region[1] * region[2];
  char *staging = (char *)malloc(count * pixel_size);
  assert(staging);
  for (size_t i = 0; i < count; ++i)
    memcpy(staging + i * pixel_size, fill_pixel, pixel_size);
  pocl_vortex_write_image_rect(data, image, mem_id, staging, NULL, origin, region,
                               0, 0, 0);
  free(staging);
  return CL_SUCCESS;
}

/* Worker thread: drains ready_list off the caller's stack so command
 * execution never re-enters a lock the submitting thread still holds. */
static void *pocl_vortex_driver_thread (void *arg) {
  vortex_device_data_t *dd = (vortex_device_data_t *)arg;

  POCL_LOCK (dd->cq_lock);
  while (1)
    {
      _cl_command_node *node;
      while ((node = dd->ready_list))
        {
          assert (pocl_command_is_ready (node->sync.event.event));
          assert (node->sync.event.event->status == CL_SUBMITTED);
          CDL_DELETE (dd->ready_list, node);
          dd->command_running = 1;
          POCL_UNLOCK (dd->cq_lock);
          pocl_exec_command (node);
          POCL_LOCK (dd->cq_lock);
          dd->command_running = 0;
        }
      /* ready_list drained; wake any flush/join waiters */
      POCL_BROADCAST_COND (dd->idle_cond);
      if (dd->exit_requested)
        break;
      POCL_WAIT_COND (dd->cq_cond, dd->cq_lock);
    }
  POCL_UNLOCK (dd->cq_lock);
  return NULL;
}

void pocl_vortex_submit (_cl_command_node *node, cl_command_queue cq) {
  vortex_device_data_t *dd = (vortex_device_data_t *)node->device->data;

  node->state = POCL_COMMAND_READY;
  POCL_LOCK (dd->cq_lock);
  pocl_command_push(node, &dd->ready_list, &dd->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  POCL_SIGNAL_COND (dd->cq_cond);
  POCL_UNLOCK (dd->cq_lock);

  return;
}

void pocl_vortex_flush (cl_device_id dev, cl_command_queue cq) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;

  POCL_LOCK (dd->cq_lock);
  POCL_SIGNAL_COND (dd->cq_cond);
  POCL_UNLOCK (dd->cq_lock);
}

void pocl_vortex_join (cl_device_id dev, cl_command_queue cq) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;

  POCL_LOCK (dd->cq_lock);
  POCL_SIGNAL_COND (dd->cq_cond);
  while (dd->ready_list != NULL || dd->command_list != NULL
         || dd->command_running)
    POCL_WAIT_COND (dd->idle_cond, dd->cq_lock);
  POCL_UNLOCK (dd->cq_lock);

  return;
}

void pocl_vortex_notify (cl_device_id dev, cl_event event, cl_event finished) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      /* Unlock the finished event to prevent a lock-order violation against
       * the command queue locked by pocl_update_event_failed. Mirror the
       * pattern used by the basic/pthread drivers. */
      pocl_unlock_events_inorder (event, finished);
      pocl_update_event_failed (CL_FAILED, NULL, 0, event, NULL);
      pocl_lock_events_inorder (finished, event);
      return;
    }

  if (node->state != POCL_COMMAND_READY)
    return;

  if (pocl_command_is_ready (event))
    {
      if (event->status == CL_QUEUED)
        {
          pocl_update_event_submitted (event);
          POCL_LOCK (dd->cq_lock);
          CDL_DELETE (dd->command_list, node);
          CDL_PREPEND (dd->ready_list, node);
          POCL_SIGNAL_COND (dd->cq_cond);
          POCL_UNLOCK (dd->cq_lock);
        }
      return;
    }
}