/* Vortex v3 KMU kernel-runtime support.
 *
 * Each OpenCL kernel is its own KMU entry point in the .vxbin: the
 * compiler glue (compile_vortex_program) emits, per kernel, an entry stub
 * + a "vortex.kernel" trampoline, and the runtime resolves each by name
 * via vx_module_get_kernel. There is no kernel-id dispatch.
 *
 * This file holds the program-wide support those trampolines and the
 * POCL-generated kernels share: the work-globals the KMU does not surface
 * via CSRs, the per-CTA local-memory accessor, and the once-per-launch
 * args prologue.
 */
#include "kernel_args.h"
#include <vx_intrinsics.h>
#include <VX_types.h>
#include <stddef.h>

/* Globals consumed by workitems.c for OpenCL functions the KMU does not
 * surface via CSRs (work_dim, global_offset). Set once per launch by
 * __vx_kernel_setup from the host-supplied kernel_args_t — same across all
 * (block, thread) invocations. */
int g_work_dim;
struct g_global_offset3 { uint32_t x, y, z; };
struct g_global_offset3 g_global_offset;

/* KMU exposes the per-CTA local-memory base address through
 * VX_CSR_CTA_LMEM_ADDR. The legacy vx_spawn.h version took a per-call size
 * parameter and did software offsetting per-workgroup; KMU centralizes the
 * address so the size argument is no longer meaningful. POCL still passes
 * one — ignored. */
void* vx_local_alloc(uint32_t size) {
  (void)size;
  return (void*)(uintptr_t)csr_read(VX_CSR_CTA_LMEM_ADDR);
}

/* Once-per-launch args prologue, called by each kernel's generated
 * trampoline. Publishes the work-globals and returns the pointer to the
 * user kernel-argument block, which follows the kernel_args_t header. */
void* __vx_kernel_setup(void) {
  kernel_args_t* kargs = (kernel_args_t*)csr_read(VX_CSR_MSCRATCH);

  g_work_dim = kargs->work_dim;
  g_global_offset.x = kargs->global_offset[0];
  g_global_offset.y = kargs->global_offset[1];
  g_global_offset.z = kargs->global_offset[2];

  uint32_t aligned_kargs_size = ALIGN_OFFSET(sizeof(kernel_args_t), sizeof(size_t));
  return (void*)((uint8_t*)kargs + aligned_kargs_size);
}

/* Dead link stub. vx_start.S's _start (the ELF ENTRY, always linked) ends
 * with `jal kernel_main`. A multi-entry .vxbin never reaches _start — the
 * KMU enters the per-kernel stubs listed in the VXSYMTAB footer — but the
 * symbol must still resolve at link time. */
void kernel_main(void) {}
