#include <stdint.h>

typedef struct {
  uint32_t work_dim;
  uint32_t num_groups[3];
  uint32_t local_size[3];
  uint32_t global_offset[3];
  uint32_t kernel_id;
} kernel_args_t;

inline uint32_t alignOffset(uint32_t offset, uint32_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}
