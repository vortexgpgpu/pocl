/* Vortex v3 KMU kernel-args layout.
 *
 * The legacy spawn_thread model carried num_groups[3]/local_size[3] in this
 * struct because the kernel-side software loop needed them. With the v3 KMU,
 * those dimensions are supplied to the dispatcher in vx_launch_info_t at
 * vx_enqueue_launch() time and surfaced inside the kernel via CSRs
 * (VX_CSR_CTA_{GRID_DIM,BLOCK_DIM}_*). We only keep the fields KMU does NOT
 * provide: the OpenCL work_dim and global_offset.
 *
 * There is no kernel-id field: each kernel is its own KMU entry point in
 * the .vxbin (resolved by name via vx_module_get_kernel), so the runtime
 * launches the kernel directly rather than dispatching on an index.
 */
#include <stdint.h>

typedef struct {
  uint32_t work_dim;
  uint32_t global_offset[3];
} kernel_args_t;

#define ALIGN_OFFSET(offset, alignment) (((offset) + (alignment) - 1) & ~((alignment) - 1))

/* Start every __local allocation on a boundary that suits any OpenCL type the
 * kernel may store there. The widest is long16/double16 at 16 * 8 = 128 bytes,
 * and the driver cannot know the pointee type of a __local pointer argument.
 * 128 also matches the local-memory bank stride, so the padding costs nothing
 * in bank-conflict terms. */
#define VX_LOCAL_ALIGN 128
