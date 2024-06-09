#include <vx_spawn.h>

#define CLK_GLOBAL_MEM_FENCE  0x02

void _Z7barrierj(int flags) {
  if (flags & CLK_GLOBAL_MEM_FENCE) {
    vx_fence();
  }
  vx_barrier(__local_group_id, __warps_per_group);
}
