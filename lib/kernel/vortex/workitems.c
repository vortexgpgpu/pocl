#include <vx_spawn.h>

extern int g_work_dim;
extern dim3_t g_global_offset;

uint32_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_work_dim (void) {
  return g_work_dim;
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_num_groups(uint32_t dimindx) {
  switch (dimindx) {
  default: return gridDim.x;
  case 1: return gridDim.y;
  case 2: return gridDim.z;
  }
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_local_size(uint32_t dimindx) {
  switch (dimindx) {
  default: return blockDim.x;
  case 1: return blockDim.y;
  case 2: return blockDim.z;
  }
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_global_offset(uint32_t dimindx) {
  switch (dimindx) {
  default: return g_global_offset.x;
  case 1: return g_global_offset.y;
  case 2: return g_global_offset.z;
  }
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_group_id(uint32_t dimindx) {
  switch (dimindx) {
  default: return blockIdx.x;
  case 1: return blockIdx.y;
  case 2: return blockIdx.z;
  }
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_local_id(uint32_t dimindx) {
  switch (dimindx) {
  default: return threadIdx.x;
  case 1: return threadIdx.y;
  case 2: return threadIdx.z;
  }
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_global_size(uint32_t dimindx) {
  switch (dimindx) {
  default: return blockDim.x * gridDim.x;
  case 1: return blockDim.y * gridDim.y;
  case 2: return blockDim.z * gridDim.z;
  }
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_global_id(uint32_t dimindx) {
  switch (dimindx) {
  default: return blockIdx.x * blockDim.x + threadIdx.x + g_global_offset.x;
  case 1: return blockIdx.y * blockDim.y + threadIdx.y + g_global_offset.y;
  case 2: return blockIdx.z * blockDim.z + threadIdx.z + g_global_offset.z;
  }
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_global_linear_id(void) {
  return ((blockIdx.z * blockDim.z + threadIdx.z) * blockDim.y * gridDim.y * blockDim.x * gridDim.x)
       + ((blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x)
       + ((blockIdx.x * blockDim.z + threadIdx.x));
}

size_t _CL_OVERLOADABLE _CL_READNONE _CL_OPTNONE
get_local_linear_id(void) {
  return (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
}