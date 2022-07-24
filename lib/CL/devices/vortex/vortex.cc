/* vortex.c - a minimalistic pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2019 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifdef __cplusplus
extern "C" {
#endif

#include "vortex.h"
#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#if !defined(OCS_AVAILABLE)
#include <vortex.h>
#endif

#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "pocl_llvm.h"

#ifdef __cplusplus
}
#endif

#include <sstream>
#include <vector>
#include <stdarg.h>

/* Maximum kernels occupancy */
#define MAX_KERNELS 16

/* default WG size in each dimension & total WG size.
 * this should be reasonable for CPU */
#define DEFAULT_WG_SIZE 4096

// location is local memory where to store kernel parameters
#define KERNEL_ARG_BASE_ADDR 0x7fff0000

// allocate 1MB OpenCL print buffer
#define PRINT_BUFFER_SIZE (1024 * 1024)

extern char *build_cflags;
extern char *build_ldflags;
extern char *build_llcflags;

struct vx_device_data_t {
#if !defined(OCS_AVAILABLE)
  vx_device_h vx_device;
  size_t vx_print_buf_d;
  vx_buffer_h vx_print_buf_h;
  uint32_t printf_buffer;
  uint32_t printf_buffer_position;   
#endif

  /* Currently loaded kernel. */
  cl_kernel current_kernel;

  /* List of commands ready to be executed */
  _cl_command_node *volatile ready_list;

  /* List of commands not yet ready to be executed */
  _cl_command_node *volatile command_list;

  /* Lock for command list related operations */
  pocl_lock_t cq_lock;  
};

struct vx_buffer_data_t {
#if !defined(OCS_AVAILABLE)
  vx_device_h vx_device;
  vx_buffer_h staging_buf;
#endif
  size_t dev_mem_addr;
};

struct kernel_context_t {
  uint32_t num_groups[3];
  uint32_t global_offset[3];
  uint32_t local_size[3];  
  uint32_t printf_buffer;
  uint32_t printf_buffer_position;
  uint32_t printf_buffer_capacity;  
  uint32_t work_dim;
};

static size_t ALIGNED_CTX_SIZE = 4 * ((sizeof(kernel_context_t) + 3) / 4);

static const cl_image_format supported_image_formats[] = {
    {CL_A, CL_SNORM_INT8},
    {CL_A, CL_SNORM_INT16},
    {CL_A, CL_UNORM_INT8},
    {CL_A, CL_UNORM_INT16},
    {CL_A, CL_SIGNED_INT8},
    {CL_A, CL_SIGNED_INT16},
    {CL_A, CL_SIGNED_INT32},
    {CL_A, CL_UNSIGNED_INT8},
    {CL_A, CL_UNSIGNED_INT16},
    {CL_A, CL_UNSIGNED_INT32},
    {CL_A, CL_FLOAT},
    {CL_RGBA, CL_SNORM_INT8},
    {CL_RGBA, CL_SNORM_INT16},
    {CL_RGBA, CL_UNORM_INT8},
    {CL_RGBA, CL_UNORM_INT16},
    {CL_RGBA, CL_SIGNED_INT8},
    {CL_RGBA, CL_SIGNED_INT16},
    {CL_RGBA, CL_SIGNED_INT32},
    {CL_RGBA, CL_UNSIGNED_INT8},
    {CL_RGBA, CL_UNSIGNED_INT16},
    {CL_RGBA, CL_UNSIGNED_INT32},
    {CL_RGBA, CL_HALF_FLOAT},
    {CL_RGBA, CL_FLOAT},
    {CL_ARGB, CL_SNORM_INT8},
    {CL_ARGB, CL_UNORM_INT8},
    {CL_ARGB, CL_SIGNED_INT8},
    {CL_ARGB, CL_UNSIGNED_INT8},
    {CL_BGRA, CL_SNORM_INT8},
    {CL_BGRA, CL_UNORM_INT8},
    {CL_BGRA, CL_SIGNED_INT8},
    {CL_BGRA, CL_UNSIGNED_INT8}};

void pocl_vortex_init_device_ops(struct pocl_device_ops *ops) {
  ops->device_name = "vortex";
  ops->probe = pocl_vortex_probe;

  ops->init = pocl_vortex_init;
  ops->reinit = NULL;
  ops->uninit = pocl_vortex_uninit;

  ops->build_hash = pocl_vortex_build_hash;

  ops->compile_kernel = pocl_vortex_compile_kernel;

#if !defined(OCS_AVAILABLE)

  ops->run = pocl_vortex_run;
  ops->run_native = NULL;

  ops->init_queue = NULL;
  ops->free_queue = NULL;

  ops->alloc_mem_obj = pocl_vortex_alloc_mem_obj;
  ops->free = pocl_vortex_free;

  ops->submit = pocl_vortex_submit;
  ops->notify = pocl_vortex_notify;
  ops->broadcast = pocl_broadcast;
  ops->join = pocl_vortex_join;
  ops->flush = pocl_vortex_flush;

  ops->wait_event = NULL;
  ops->update_event = NULL;
  ops->free_event_data = NULL;

  ops->map_mem = NULL;
  ops->unmap_mem = NULL;

  ops->read = pocl_vortex_read;
  ops->write = pocl_vortex_write;

  ops->read_rect = NULL;
  ops->write_rect = NULL;

  ops->copy = NULL;
  ops->copy_rect = NULL;

  ops->svm_free = NULL;
  ops->svm_alloc = NULL;
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = NULL;
  ops->svm_fill = NULL;

#endif
}

char *pocl_vortex_build_hash(cl_device_id device) {
  char *res = (char *)calloc(1000, sizeof(char));
  snprintf(res, 1000, "vortex-riscv32-unknown-unknown-elf");
  return res;
}

static cl_device_partition_property vortex_partition_properties[1] = {0};

static const char *final_ld_flags[] = {"-nostartfiles", HOST_LD_FLAGS_ARRAY, NULL};

unsigned int pocl_vortex_probe(struct pocl_device_ops *ops) {
  if (0 == strcmp(ops->device_name, "vortex"))
    return 1;
  return 0;
}

cl_int pocl_vortex_init(unsigned j, cl_device_id device,
                        const char *parameters) {
  struct vx_device_data_t *d;
  cl_int ret = CL_SUCCESS;
  int err;
  static int first_vortex_init = 1;

  if (first_vortex_init) {
    POCL_MSG_WARN("INIT dlcache DOTO delete\n");
    pocl_init_dlhandle_cache();
    first_vortex_init = 0;
  }
  device->global_mem_id = 0;

  pocl_init_cpu_device_infos(device);

  err = pocl_topology_detect_device_info(device);
  if (err != 0)
    return CL_INVALID_DEVICE;

  d = (struct vx_device_data_t *)calloc(1, sizeof(struct vx_device_data_t));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device); 

  device->vendor = "Georgia Tech";
  device->vendor_id = 0;
  device->type = CL_DEVICE_TYPE_GPU;

  device->address_bits = 32;  
  device->llvm_target_triplet = "riscv32";
  device->llvm_cpu = "";

  device->max_compute_units = 1;

#if !defined(OCS_AVAILABLE)
  vx_device_h vx_device;

  err = vx_dev_open(&vx_device);
  if (err != 0) {
    free(d);
    return CL_DEVICE_NOT_FOUND;
  }
  
  device->device_side_printf = 1;
  device->printf_buffer_size = PRINT_BUFFER_SIZE;

  // add storage for position pointer
  uint32_t print_buf_dev_size = PRINT_BUFFER_SIZE + sizeof(uint32_t);

  size_t vx_print_buf_d;
  err = vx_mem_alloc(vx_device, print_buf_dev_size, &vx_print_buf_d);
  if (err != 0) {
    vx_dev_close(vx_device);
    free(d);
    return CL_INVALID_DEVICE;
  }  
  
  vx_buffer_h vx_print_buf_h;
  err = vx_buf_alloc(vx_device, print_buf_dev_size, &vx_print_buf_h);
  if (err != 0) {
    vx_dev_close(vx_device);
    free(d);
    return CL_OUT_OF_HOST_MEMORY;
  }  
    
  // clear print position to zero
  uint8_t* staging_ptr = (uint8_t*)vx_host_ptr(vx_print_buf_h);
  memset(staging_ptr + PRINT_BUFFER_SIZE, 0, sizeof(uint32_t));
  err = vx_copy_to_dev(vx_print_buf_h, vx_print_buf_d + PRINT_BUFFER_SIZE, sizeof(uint32_t), PRINT_BUFFER_SIZE);
  if (err != 0) {
    vx_buf_free(vx_print_buf_h);
    vx_dev_close(vx_device);
    free(d);
    return CL_OUT_OF_HOST_MEMORY;
  }
    
  d->vx_device      = vx_device;
  d->vx_print_buf_d = vx_print_buf_d;
  d->vx_print_buf_h = vx_print_buf_h; 
  d->printf_buffer  = vx_print_buf_d;
  d->printf_buffer_position = vx_print_buf_d + PRINT_BUFFER_SIZE;
#endif 

  d->current_kernel = NULL;

  POCL_INIT_LOCK(d->cq_lock);

  device->data = d;

  return ret;
}

cl_int pocl_vortex_uninit(unsigned j, cl_device_id device) {
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;
  if (NULL == d)
    return CL_SUCCESS;  

#if !defined(OCS_AVAILABLE)
  vx_buf_free(d->vx_print_buf_h);
  vx_dev_close(d->vx_device);
#endif

  POCL_DESTROY_LOCK(d->cq_lock);
  POCL_MEM_FREE(d);
  device->data = NULL;
  
  return CL_SUCCESS;
}

#if !defined(OCS_AVAILABLE)

static void *
pocl_vortex_malloc(cl_device_id device, cl_mem_flags flags, size_t size,
                   void *host_ptr) {
  auto d = (vx_device_data_t *)device->data;
  void *b = NULL;
  pocl_global_mem_t *mem = device->global_memory;
  int err;

  if (flags & CL_MEM_USE_HOST_PTR) {
    std::abort(); //TODO
  }

  vx_buffer_h staging_buf;
  err = vx_buf_alloc(d->vx_device, size, &staging_buf);
  if (err != 0)
    return nullptr;

  size_t dev_mem_addr;
  err = vx_mem_alloc(d->vx_device, size, &dev_mem_addr);
  if (err != 0) {
    vx_buf_free(staging_buf);
    return nullptr;
  }

  if (flags & CL_MEM_COPY_HOST_PTR) {
    auto buf_ptr = vx_host_ptr(staging_buf);
    memcpy((void*)buf_ptr, host_ptr, size);
    err = vx_copy_to_dev(staging_buf, dev_mem_addr, size, 0);
    if (err != 0) {
      vx_buf_free(staging_buf);
      return nullptr;
    }
  }

  auto buf_data = new vx_buffer_data_t();
  buf_data->vx_device    = d->vx_device;
  buf_data->staging_buf  = staging_buf;
  buf_data->dev_mem_addr = dev_mem_addr;

  return buf_data;
}

cl_int
pocl_vortex_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void *host_ptr) {
  void *b = NULL;
  cl_mem_flags flags = mem_obj->flags;
  unsigned i;

  /* Check if some driver has already allocated memory for this mem_obj
     in our global address space, and use that. */
  for (i = 0; i < mem_obj->context->num_devices; ++i) {
    if (!mem_obj->device_ptrs[i].available)
      continue;
    if (mem_obj->device_ptrs[i].global_mem_id == device->global_mem_id && mem_obj->device_ptrs[i].mem_ptr != NULL) {
      mem_obj->device_ptrs[device->dev_id].mem_ptr = mem_obj->device_ptrs[i].mem_ptr;
      //vortex_memory_register(mem_obj->device_ptrs[device->dev_id].mem_ptr, mem_obj->size);
      POCL_MSG_PRINT_INFO("VORTEX: alloc_mem_obj, use already allocated memory\n");
      std::abort(); // TODO
      return CL_SUCCESS;
    }
  }

  /* Memory for this global memory is not yet allocated -> we'll allocate it. */
  b = pocl_vortex_malloc(device, flags, mem_obj->size, host_ptr);
  if (b == NULL)
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  /* Take ownership if not USE_HOST_PTR. */
  if (~flags & CL_MEM_USE_HOST_PTR)
    mem_obj->shared_mem_allocation_owner = device;

  mem_obj->device_ptrs[device->dev_id].mem_ptr = b;

  if (flags & CL_MEM_ALLOC_HOST_PTR) {
    std::abort(); // TODO
  }

  return CL_SUCCESS;
}

void pocl_vortex_free(cl_device_id device, cl_mem memobj) {
  cl_mem_flags flags = memobj->flags;
  auto buf_data = (vx_buffer_data_t*)memobj->device_ptrs[device->dev_id].mem_ptr;

  if (flags & CL_MEM_USE_HOST_PTR 
   || memobj->shared_mem_allocation_owner != device) {
    std::abort(); //TODO
  } else {
    vx_buf_free(buf_data->staging_buf);
    vx_mem_free(buf_data->vx_device, buf_data->dev_mem_addr);
  }
  if (memobj->flags | CL_MEM_ALLOC_HOST_PTR)
    memobj->mem_host_ptr = NULL;
}

static void vortex_command_scheduler(struct vx_device_data_t *d) {
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = d->ready_list)) {
    assert(pocl_command_is_ready(node->event));
    assert(node->event->status == CL_SUBMITTED);
    CDL_DELETE(d->ready_list, node);
    POCL_UNLOCK(d->cq_lock);
    pocl_exec_command(node);
    POCL_LOCK(d->cq_lock);
  }

  return;
}

void pocl_vortex_submit(_cl_command_node *node, cl_command_queue cq) {
  struct vx_device_data_t *d = (struct vx_device_data_t *)node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache(node, 1, 1);

  node->ready = 1;
  POCL_LOCK(d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ(node->event);
  vortex_command_scheduler(d);
  POCL_UNLOCK(d->cq_lock);

  return;
}

void pocl_vortex_flush(cl_device_id device, cl_command_queue cq) {
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;

  POCL_LOCK(d->cq_lock);
  vortex_command_scheduler(d);
  POCL_UNLOCK(d->cq_lock);
}

void pocl_vortex_join(cl_device_id device, cl_command_queue cq) {
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;

  POCL_LOCK(d->cq_lock);
  vortex_command_scheduler(d);
  POCL_UNLOCK(d->cq_lock);

  return;
}

void pocl_vortex_notify(cl_device_id device, cl_event event, cl_event finished) {
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;
  _cl_command_node *volatile node = event->command;

  if (finished->status < CL_COMPLETE) {
    pocl_update_event_failed(event);
    return;
  }

  if (!node->ready)
    return;

  if (pocl_command_is_ready(event)) {
    if (event->status == CL_QUEUED) {
      pocl_update_event_submitted(event);
      POCL_LOCK(d->cq_lock);
      CDL_DELETE(d->command_list, node);
      CDL_PREPEND(d->ready_list, node);
      vortex_command_scheduler(d);
      POCL_UNLOCK(d->cq_lock);
    }
    return;
  }
}

void pocl_vortex_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset, 
                       size_t size) {
  auto buf_data = (vx_buffer_data_t*)dst_mem_id->mem_ptr;
  auto buf_ptr = vx_host_ptr(buf_data->staging_buf);
  memcpy((char *)buf_ptr + offset, host_ptr, size);
  auto vx_err = vx_copy_to_dev(buf_data->staging_buf, buf_data->dev_mem_addr, offset + size, 0);
  assert(0 == vx_err);
}

void pocl_vortex_read(void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t offset, 
                      size_t size) {
  int vx_err;
  struct vx_device_data_t *d = (struct vx_device_data_t *)data;                      
  auto buf_data = (vx_buffer_data_t*)src_mem_id->mem_ptr;
  vx_err = vx_copy_from_dev(buf_data->staging_buf, buf_data->dev_mem_addr, offset + size, 0);
  assert(0 == vx_err);
  auto buf_ptr = vx_host_ptr(buf_data->staging_buf);
  assert(buf_ptr);
  memcpy(host_ptr, (char *)buf_ptr + offset, size);
}

static void print_data(const char* text, const uint8_t* data, size_t size) {
  /*printf(text);
  for (size_t i = 0; i < size; ++i)
    printf("%02x", data[size-1-i]);
  printf("\n");*/
}

void pocl_vortex_run(void *data, _cl_command_node *cmd) {
  struct vx_device_data_t *d;
  size_t x, y, z;
  unsigned i;
  unsigned dev_i = cmd->device_i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;
  int err;
  
  assert(data != NULL);
  d = (struct vx_device_data_t *)data;

  // calculate kernel arguments buffer size
  size_t abuf_size = 0;  
  size_t abuf_args_size = 4 * (meta->num_args + meta->num_locals);
  size_t abuf_ext_size = 0;
  {
    // pocl_context data
    abuf_size += ALIGNED_CTX_SIZE; 
    // argument data    
    abuf_size += abuf_args_size;
    for (i = 0; i < meta->num_args; ++i) {  
      auto al = &(cmd->command.run.arguments[i]);  
      if (ARG_IS_LOCAL(meta->arg_info[i])
       && !cmd->device->device_alloca_locals) {
        abuf_size += 4;
        abuf_size += al->size;
        abuf_ext_size += al->size;
      } else
      if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
       || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
        abuf_size += 4;
      } else {
        abuf_size += al->size;
      }
    }
  }
  assert(abuf_size <= 0xffff);

  // allocate kernel arguments buffer
  vx_buffer_h staging_buf;
  err = vx_buf_alloc(d->vx_device, abuf_size, &staging_buf);
  assert(0 == err);

  // update kernel arguments buffer
  {
    auto abuf_ptr = (uint8_t*)vx_host_ptr(staging_buf);
    assert(abuf_ptr);

    // write context data
    {
      kernel_context_t ctx;
      for (int i = 0; i < 3; ++i) {
        ctx.num_groups[i] = pc->num_groups[i];
        ctx.global_offset[i] = pc->global_offset[i];
        ctx.local_size[i] = pc->local_size[i];        
      }
      ctx.work_dim = pc->work_dim;      
      ctx.printf_buffer = d->printf_buffer;
      ctx.printf_buffer_position = d->printf_buffer_position;
      ctx.printf_buffer_capacity = PRINT_BUFFER_SIZE;

      memset(abuf_ptr, 0, ALIGNED_CTX_SIZE);
      memcpy(abuf_ptr, &ctx, sizeof(kernel_context_t));
      print_data("*** ctx=", abuf_ptr, ALIGNED_CTX_SIZE);
    }

    // write arguments    
    uint32_t args_base_addr = KERNEL_ARG_BASE_ADDR;
    uint32_t args_addr = args_base_addr + ALIGNED_CTX_SIZE + abuf_args_size;
    uint32_t args_ext_addr = (args_base_addr + abuf_size) - abuf_ext_size;
    for (i = 0; i < meta->num_args; ++i) {
      uint32_t addr = ALIGNED_CTX_SIZE + i * 4;
      auto al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i])) {
        if (cmd->device->device_alloca_locals) {
          memcpy(abuf_ptr + addr, &al->size, 4);
          print_data("*** locals=", abuf_ptr + addr, 4);
        } else {
          memcpy(abuf_ptr + addr, &args_addr, 4);          
          memcpy(abuf_ptr + (args_addr - args_base_addr), &args_ext_addr, 4);
          args_addr += 4;
          args_ext_addr += al->size;
          std::abort();
        }
      } else
      if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
        memcpy(abuf_ptr + addr, &args_addr, 4);        
        if (al->value == NULL) {
          memset(abuf_ptr + (args_addr - args_base_addr), 0, 4);
          print_data("*** null=", abuf_ptr + (args_addr - args_base_addr), 4); 
        } else {
          cl_mem m = (*(cl_mem *)(al->value));
          auto buf_data = (vx_buffer_data_t*)m->device_ptrs[cmd->device->dev_id].mem_ptr;
          auto dev_mem_addr = buf_data->dev_mem_addr + al->offset;
          memcpy(abuf_ptr + (args_addr - args_base_addr), &dev_mem_addr, 4);
          print_data("*** ptr=", abuf_ptr + (args_addr - args_base_addr), 4);
        }
        args_addr += 4;
      } else 
      if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
        std::abort();
      } else 
      if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
        std::abort();
      } else {
        memcpy(abuf_ptr + addr, &args_addr, 4);
        memcpy(abuf_ptr + (args_addr - args_base_addr), al->value, al->size);
        print_data("*** arg-addr=", abuf_ptr + addr, 4);
        print_data("*** arg-value=", abuf_ptr + (args_addr - args_base_addr), al->size);
        args_addr += al->size;
      }
    }

    // upload kernel arguments buffer
    err = vx_copy_to_dev(staging_buf, args_base_addr, abuf_size, 0);
    assert(0 == err);

    // release staging buffer
    err = vx_buf_free(staging_buf);
    assert(0 == err);
    
    // upload kernel to device
    if (NULL == d->current_kernel 
     || d->current_kernel != kernel) {    
       d->current_kernel = kernel;
      char program_bin_path[POCL_FILENAME_LENGTH];
      pocl_cache_final_binary_path (program_bin_path, program, dev_i, kernel, NULL, 0);
      err = vx_upload_kernel_file(d->vx_device, program_bin_path);      
      assert(0 == err);
    }
  }
    
  // quick off kernel execution
  err = vx_start(d->vx_device);
  assert(0 == err);

  // wait for the execution to complete
  err = vx_ready_wait(d->vx_device, -1);
  assert(0 == err);

  // flush print buffer 
  {
    auto print_ptr = (uint8_t*)vx_host_ptr(d->vx_print_buf_h);
    err = vx_copy_from_dev(d->vx_print_buf_h, d->vx_print_buf_d + PRINT_BUFFER_SIZE, sizeof(uint32_t), PRINT_BUFFER_SIZE);
    assert(0 == err);
    uint32_t print_size = *(uint32_t*)(print_ptr + PRINT_BUFFER_SIZE);
    if (print_size != 0) {
      err = vx_copy_from_dev(d->vx_print_buf_h, d->vx_print_buf_d, print_size, 0);
      assert(0 == err);      
      
      write (STDOUT_FILENO, print_ptr, print_size);
      
      memset(print_ptr + PRINT_BUFFER_SIZE, 0, sizeof(uint32_t));
      err = vx_copy_to_dev(d->vx_print_buf_h, d->vx_print_buf_d, sizeof(uint32_t), PRINT_BUFFER_SIZE);
      assert(0 == err);
    }
  }

  pocl_release_dlhandle_cache(cmd);
}
#endif

void pocl_vortex_compile_kernel(_cl_command_node *cmd, cl_kernel kernel,
                                cl_device_id device, int specialize) {
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache(cmd, 0, specialize);
}

#if defined(OCS_AVAILABLE)

class StaticStrFormat {
public:

  StaticStrFormat(size_t size) : store_(size), index_(0) {}

  const char* format(const char* format, ...) {
    auto & buffer = store_.at(index_++);

    va_list args_orig, args_copy;
    va_start(args_orig, format);

    va_copy(args_copy, args_orig);
    size_t size = vsnprintf(nullptr, 0, format, args_copy) + 1;
    va_end(args_copy);

    buffer.resize(size);

    vsnprintf(buffer.data(), size, format, args_orig);       
    va_end(args_orig);

    return buffer.data();
  }

private:
  std::vector<std::vector<char>> store_;
  size_t index_;
};

int exec(const char* cmd, std::ostream& out) {  
  char buffer[128];
  auto pipe = popen(cmd, "r");
  if (!pipe) {
      throw std::runtime_error("popen() failed!");
  }
  while (!feof(pipe)) {
      if (fgets(buffer, 128, pipe) != nullptr)
          out << buffer;
  }
  return pclose(pipe);
}

int pocl_llvm_build_vortex_program(cl_kernel kernel, 
                                   unsigned device_i, 
                                   cl_device_id device,
                                   const char *kernel_bc,
                                   const char *kernel_obj,
                                   char *kernel_out) {
  char kernel_elf[POCL_FILENAME_LENGTH];
  cl_program program = kernel->program;
  int err;

  const char* llvm_install_path = getenv("LLVM_PREFIX");
  if (llvm_install_path) {
    if (!pocl_exists(llvm_install_path)) {
      POCL_MSG_ERR("$LLVM_PREFIX: '%s' doesn't exist\n", llvm_install_path);
      return -1;    
    }
    POCL_MSG_PRINT_INFO("using $LLVM_PREFIX=%s!\n", llvm_install_path);
  }

  {
    std::string llc_path(LLVM_LLC);
    if (llvm_install_path) {
      llc_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
    }

    std::stringstream ss_cmd, ss_out;
    ss_cmd << llc_path.c_str() << " " << build_llcflags << " -filetype=obj -o " << kernel_obj << " " << kernel_bc;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }

    /*
    ss_cmd.str("");
    ss_cmd << llc_path.c_str() << " " << build_llcflags << " -filetype=asm -o " << kernel->name << "_kernel.asm" << " " << kernel_bc;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
    */
  }
  
  {  
    char wrapper_cc[POCL_FILENAME_LENGTH];    
    char pfn_workgroup_string[WORKGROUP_STRING_LENGTH];
    std::stringstream ss;

    snprintf (pfn_workgroup_string, WORKGROUP_STRING_LENGTH,
              "_pocl_kernel_%s_workgroup", kernel->name);
    
    ss << "#include <vx_spawn.h>\n"
          "void " << pfn_workgroup_string << "(uint8_t* args, uint8_t*, uint32_t, uint32_t, uint32_t);\n"  
          "int main() {\n"
          "  const context_t* ctx = (const context_t*)" << KERNEL_ARG_BASE_ADDR << ";\n"
          "  void* args = (void*)" << (KERNEL_ARG_BASE_ADDR + ALIGNED_CTX_SIZE) << ";\n"
          "  vx_spawn_kernel(ctx, (void*)" << pfn_workgroup_string << ", args);\n"
          "  return 0;\n"
          "}";

    auto content = ss.str();
    err = pocl_write_tempfile(wrapper_cc, "/tmp/pocl_vortex_kernel", ".c",
                              content.c_str(), content.size(), nullptr);
    if (err != 0)
      return err;
 
    err = pocl_mk_tempname(kernel_elf, "/tmp/pocl_vortex_kernel", ".elf", nullptr);
    if (err != 0)
      return err;

    StaticStrFormat ssfmt(9);

    std::string clang_path(CLANG);
    if (llvm_install_path) {
      clang_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
    }

    {
      std::stringstream ss_cmd, ss_out;
      ss_cmd << clang_path.c_str() << " " << build_cflags << " -I" POCL_INSTALL_PRIVATE_HEADER_DIR << " "<< wrapper_cc << " " << kernel_obj << " " << build_ldflags << " -o " << kernel_elf;
      POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
      err = exec(ss_cmd.str().c_str(), ss_out);
      if (err != 0) {
        POCL_MSG_ERR("%s\n", ss_out.str().c_str());
        return err;
      }
    }
  }

  {
    std::string objcopy_path(LLVM_OBJCOPY);
    if (llvm_install_path) {
      objcopy_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
    }

    std::stringstream ss_cmd, ss_out;
    ss_cmd << objcopy_path.c_str() << " -O binary " << kernel_elf << " " << kernel_out;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  {
    std::string objdump_path(LLVM_OBJDUMP);
    if (llvm_install_path) {
      objdump_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
    }

    std::stringstream ss_cmd, ss_out;
    ss_cmd << objdump_path.c_str() << " -arch=riscv32 -mcpu=generic-rv32 -mattr=+m,+f -mattr=+vortex -D " << kernel_elf << " > " << kernel->name << ".dump";
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }
  
  return 0;
}

#endif