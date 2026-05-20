/* Vortex v3 KMU kernel-args layout.
 *
 * The legacy spawn_thread model carried num_groups[3]/local_size[3] in this
 * struct because the kernel-side software loop needed them. With the v3 KMU,
 * those dimensions are supplied to the dispatcher at vx_start_g() time and
 * surfaced inside the kernel via CSRs (VX_CSR_CTA_{GRID_DIM,BLOCK_DIM}_*).
 * We only keep the fields KMU does NOT provide: the OpenCL work_dim,
 * global_offset, and the kernel-id lookup index. See
 * pocl_vortex_v3_proposal.md §3.2 (Phase 2).
 */
#include <stdint.h>

typedef struct {
  uint32_t work_dim;
  uint32_t global_offset[3];
  uint32_t kernel_id;
} kernel_args_t;

#define ALIGN_OFFSET(offset, alignment) (((offset) + (alignment) - 1) & ~((alignment) - 1))
