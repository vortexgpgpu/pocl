/* OpenCL work-item helpers, KMU edition.
 *
 * Legacy implementation read TLS variables (blockIdx, threadIdx, ...) that
 * vx_spawn_threads() set per-software-thread. v3 KMU surfaces the same
 * state through CSRs the hardware sets per-invocation; the OpenCL
 * builtins below just read those CSRs.
 *
 * Mirrors lib/CL/devices/vortex/kernel_main.c (Phase 2 of
 * pocl_vortex_v3_proposal.md).
 */
#include <vx_intrinsics.h>
#include <VX_types.h>

#if __riscv_xlen == 64
    typedef uint64_t SizeT;
#elif __riscv_xlen == 32
    typedef uint32_t SizeT;
#else
    #error "Unsupported RISC-V XLEN"
#endif

extern int g_work_dim;
struct g_global_offset3 { uint32_t x, y, z; };
extern struct g_global_offset3 g_global_offset;

#define _CL_OVERLOADABLE __attribute__((overloadable))

/* --- the KMU-owned dimensions (read from CSRs) --- */

static inline __attribute__((always_inline)) uint32_t kmu_thread_id(uint32_t dim) {
  switch (dim) {
    default: return (uint32_t)csr_read(VX_CSR_CTA_THREAD_ID_X);
    case 1:  return (uint32_t)csr_read(VX_CSR_CTA_THREAD_ID_Y);
    case 2:  return (uint32_t)csr_read(VX_CSR_CTA_THREAD_ID_Z);
  }
}

static inline __attribute__((always_inline)) uint32_t kmu_block_id(uint32_t dim) {
  switch (dim) {
    default: return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_X);
    case 1:  return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_Y);
    case 2:  return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_ID_Z);
  }
}

static inline __attribute__((always_inline)) uint32_t kmu_block_dim(uint32_t dim) {
  switch (dim) {
    default: return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_X);
    case 1:  return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_Y);
    case 2:  return (uint32_t)csr_read(VX_CSR_CTA_BLOCK_DIM_Z);
  }
}

static inline __attribute__((always_inline)) uint32_t kmu_grid_dim(uint32_t dim) {
  switch (dim) {
    default: return (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_X);
    case 1:  return (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_Y);
    case 2:  return (uint32_t)csr_read(VX_CSR_CTA_GRID_DIM_Z);
  }
}

/* --- OpenCL builtins --- */

uint32_t _CL_OVERLOADABLE
get_work_dim (void) {
  return g_work_dim;
}

SizeT _CL_OVERLOADABLE
get_num_groups(uint32_t dimindx) {
  return kmu_grid_dim(dimindx);
}

SizeT _CL_OVERLOADABLE
get_local_size(uint32_t dimindx) {
  return kmu_block_dim(dimindx);
}

SizeT _CL_OVERLOADABLE
get_global_offset(uint32_t dimindx) {
  switch (dimindx) {
    default: return g_global_offset.x;
    case 1:  return g_global_offset.y;
    case 2:  return g_global_offset.z;
  }
}

SizeT _CL_OVERLOADABLE
get_group_id(uint32_t dimindx) {
  return kmu_block_id(dimindx);
}

SizeT _CL_OVERLOADABLE
get_local_id(uint32_t dimindx) {
  return kmu_thread_id(dimindx);
}

SizeT _CL_OVERLOADABLE
get_global_size(uint32_t dimindx) {
  return (SizeT)kmu_block_dim(dimindx) * (SizeT)kmu_grid_dim(dimindx);
}

SizeT _CL_OVERLOADABLE
get_global_id(uint32_t dimindx) {
  SizeT base = (SizeT)kmu_block_id(dimindx) * (SizeT)kmu_block_dim(dimindx)
             + (SizeT)kmu_thread_id(dimindx);
  switch (dimindx) {
    default: return base + g_global_offset.x;
    case 1:  return base + g_global_offset.y;
    case 2:  return base + g_global_offset.z;
  }
}

SizeT _CL_OVERLOADABLE
get_global_linear_id(void) {
  uint32_t bx = kmu_block_dim(0), gx = kmu_grid_dim(0);
  uint32_t by = kmu_block_dim(1), gy = kmu_grid_dim(1);
  uint32_t bz = kmu_block_dim(2);
  uint32_t tx = kmu_thread_id(0),  ty = kmu_thread_id(1),  tz = kmu_thread_id(2);
  uint32_t Bx = kmu_block_id(0),   By = kmu_block_id(1),   Bz = kmu_block_id(2);

  return ((SizeT)(Bz * bz + tz) * by * gy * bx * gx)
       + ((SizeT)(By * by + ty) * bx * gx)
       + ((SizeT)(Bx * bx + tx));
}

SizeT _CL_OVERLOADABLE
get_local_linear_id(void) {
  uint32_t bx = kmu_block_dim(0);
  uint32_t by = kmu_block_dim(1);
  return ((SizeT)kmu_thread_id(2) * by * bx)
       + ((SizeT)kmu_thread_id(1) * bx)
       +  (SizeT)kmu_thread_id(0);
}
