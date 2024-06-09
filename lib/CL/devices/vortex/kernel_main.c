#include <vx_spawn.h>
#include <vx_print.h>
#include "kernel_args.h"

int g_work_dim;
dim3_t g_global_offset;

void* vx_local_alloc(uint32_t size) {
  return __local_mem(size);
}

void* __vx_get_kernel_callback(int kernel_id);

int main(void) {
  kernel_args_t* kargs = (kernel_args_t*)csr_read(VX_CSR_MSCRATCH);

  g_work_dim = kargs->work_dim;
  for (int i = 0, n = kargs->work_dim; i < 3; i++) {
    g_global_offset.m[i] = (i < n) ? kargs->global_offset[i] : 0;
  }

  void* arg = (void*)((uint8_t*)kargs + sizeof(kernel_args_t));
  vx_kernel_func_cb kernel_func = (vx_kernel_func_cb)__vx_get_kernel_callback(kargs->kernel_id);
  return vx_spawn_threads(kargs->work_dim, kargs->num_groups, kargs->local_size, kernel_func, arg);
}
