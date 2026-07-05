/* OpenCL work-group barrier, KMU edition.
 *
 * Legacy implementation read the spawn_thread TLS globals
 * (__local_group_id / __warps_per_group) that vx_spawn_threads() set.
 * Under v3 KMU dispatch those are never initialized, so vx_barrier()
 * was called with a count of 0 and simx aborted ("barrier_arrive
 * count=0"). The KMU exposes the same state via CSRs; mirror
 * sw/kernel/include/vx_spawn2.h's __syncthreads().
 *
 * NB: POCL upstream 0b52bd97e ("disable optimization when building
 * builtin library", Oct 2024 -> POCL 7.0) hardcoded -O0 in
 * cmake/bitcode_rules.cmake, so vx_intrinsics.h's plain `inline void
 * vx_barrier()` / `vx_fence()` no longer get inlined and end up as
 * unresolved declarations in kernel-riscv*.bc. Emit the asm directly
 * here -- the BC linker has no other way to resolve them since the
 * intrinsics live in headers, not a .so.
 */
#include <VX_types.h>

#define CLK_GLOBAL_MEM_FENCE  0x02
#define RISCV_CUSTOM0         0x0B

static __attribute__((always_inline)) inline void vx_fence_(void) {
  __asm__ volatile ("fence iorw, iorw" ::: "memory");
}

static __attribute__((always_inline)) inline void
vx_barrier_(int barrier_id, int num_warps) {
  __asm__ volatile (".insn r %0, 4, 0, x0, %1, %2"
                    :: "i"(RISCV_CUSTOM0), "r"(barrier_id), "r"(num_warps)
                    : "memory");
}

static __attribute__((always_inline)) inline unsigned long
csr_read_(int csr) {
  unsigned long v;
  __asm__ volatile ("csrr %0, %1" : "=r"(v) : "i"(csr));
  return v;
}

/* barrier() is a cross-warp convergence point: the optimizer must not duplicate
 * it onto a thread-divergent path. At -O3 the loop optimizer otherwise peels a
 * barrier-carrying loop and leaves a duplicated vx_bar guarded by a per-thread
 * predicate (e.g. a get_local_id-derived mask), so the barrier waits on warps
 * that never arrive and the work-group deadlocks (pathfinder/hotspot/dwt2d/
 * b+tree). `convergent` is the intended marker but LLVM's attribute inference
 * strips it at -O3 (the inline vx_bar asm is not itself convergent); `noduplicate`
 * survives -O3 and is what actually forbids the peel/unroll duplication, and
 * `noinline` keeps barrier() a single call the loop optimizer treats atomically.
 * The Vortex hardware barrier itself is correct. */
__attribute__((convergent, noduplicate, noinline))
void _Z7barrierj(int flags) {
  if (flags & CLK_GLOBAL_MEM_FENCE) {
    vx_fence_();
  }
  vx_barrier_((int)csr_read_(VX_CSR_CTA_ID),
              (int)csr_read_(VX_CSR_CTA_SIZE));
}

/* OpenCL mem_fence / read_mem_fence / write_mem_fence.
 *
 * The frontend lowers mem_fence(flags) to _cl_mem_fence(uint) (and the
 * read/write variants). POCL's generic lib/kernel/mem_fence.c is not part
 * of the Vortex kernel library, so these symbols were unresolved
 * (e.g. "Cannot find symbol _Z13_cl_mem_fencej in kernel library").
 * A full RISC-V fence provides the required ordering for both local and
 * global memory; emit it directly here for the same -O0 reason as the
 * barrier above.
 */
void _Z13_cl_mem_fencej(unsigned int flags) {
  (void)flags;
  vx_fence_();
}

void _Z18_cl_read_mem_fencej(unsigned int flags) {
  (void)flags;
  vx_fence_();
}

void _Z19_cl_write_mem_fencej(unsigned int flags) {
  (void)flags;
  vx_fence_();
}
