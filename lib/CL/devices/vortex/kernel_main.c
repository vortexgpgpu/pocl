/* Vortex v3 KMU kernel trampoline.
 *
 * The legacy model wrapped this with a vx_spawn_threads() software loop:
 * a single hardware thread bootstrapped main(), then iterated through every
 * (block, thread) coordinate setting TLS variables and dispatching to the
 * user kernel each time. v3 KMU handles that iteration in hardware — main()
 * is called once per (block, thread), with the per-coordinate state already
 * exposed in CSRs (VX_CSR_CTA_THREAD_ID_*, VX_CSR_CTA_BLOCK_ID_*, etc).
 *
 * This file is now a thin dispatch: read the kargs pointer from MSCRATCH,
 * extract OpenCL globals (work_dim, global_offset) into static state that
 * workitems.c references, then call into the POCL-generated kernel
 * wrapper. See pocl_vortex_v3_proposal.md §3.2 (Phase 2).
 */
#include "kernel_args.h"
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <VX_types.h>

/* Globals consumed by workitems.c for OpenCL functions that the KMU does
 * not surface via CSRs (work_dim, global_offset). They are set once per
 * launch from the host-supplied kernel_args_t — same across all
 * (block, thread) invocations. */
int g_work_dim;
struct g_global_offset3 { uint32_t x, y, z; };
struct g_global_offset3 g_global_offset;

/* KMU exposes the per-CTA local-memory base address through
 * VX_CSR_CTA_LMEM_ADDR (sw/kernel/include/vx_spawn2.h does the same as
 * `#define __local_mem() (void*)(csr_read(VX_CSR_CTA_LMEM_ADDR))`).
 * The legacy vx_spawn.h version took a per-call size parameter and did
 * software offsetting per-workgroup; KMU centralizes the address so the
 * size argument is no longer meaningful. POCL still passes one — ignored. */
void* vx_local_alloc(uint32_t size) {
  (void)size;
  return (void*)(uintptr_t)csr_read(VX_CSR_CTA_LMEM_ADDR);
}

typedef void (*vx_kernel_func_cb)(void *arg);
void* __vx_get_kernel_callback(int kernel_id);

/* KMU entry point. libvortex2.a's vx_start.S calls this symbol once per
 * (block, thread) coordinate after setting the per-invocation CSRs
 * (THREAD_ID, BLOCK_ID, BLOCK_DIM, GRID_DIM, CTA_LMEM_ADDR). The
 * vortex.kernel annotation marks it for the Vortex-specific compiler
 * pass; see sw/kernel/include/vx_spawn2.h `__kernel` macro. */
__attribute__((annotate("vortex.kernel")))
void kernel_main(void) {
  kernel_args_t* kargs = (kernel_args_t*)csr_read(VX_CSR_MSCRATCH);

  g_work_dim = kargs->work_dim;
  g_global_offset.x = kargs->global_offset[0];
  g_global_offset.y = kargs->global_offset[1];
  g_global_offset.z = kargs->global_offset[2];

  uint32_t aligned_kargs_size = ALIGN_OFFSET(sizeof(kernel_args_t), sizeof(size_t));
  void* user_args = (void*)((uint8_t*)kargs + aligned_kargs_size);
  vx_kernel_func_cb kernel_func = (vx_kernel_func_cb)__vx_get_kernel_callback(kargs->kernel_id);
  kernel_func(user_args);
}
