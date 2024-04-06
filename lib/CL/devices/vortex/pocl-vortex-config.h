#ifndef POCL_VORTEX_CONFIG
#define POCL_VORTEX_CONFIG

#include <stdint.h>
#include <string.h>

//static char *vortex_build_cflags = ;
//static char *vortex_build_ldflags;

#define WORKGROUP_STRING_LENGTH 1024
#define KERNEL_ARG_BASE_ADDR 0x7fff0000

typedef struct {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];
  uint32_t printf_buffer;
  uint32_t printf_buffer_position;
  uint32_t printf_buffer_capacity;
  uint32_t work_dim;
} pocl_kernel_context_t;

static size_t ALIGNED_CTX_SIZE = 4 * ((sizeof(pocl_kernel_context_t) + 3) / 4);

#endif
