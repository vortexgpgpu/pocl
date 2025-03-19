#include <stdint.h>

typedef struct {
  uint32_t work_dim;
  uint32_t num_groups[3];
  uint32_t local_size[3];
  uint32_t global_offset[3];
  uint32_t kernel_id;
} kernel_args_t;

#define ALIGN_OFFSET(offset, alignment) (((offset) + (alignment) - 1) & ~((alignment) - 1))