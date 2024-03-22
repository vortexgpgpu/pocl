/* vortex.c - a minimalistic single core pocl device driver layer implementation

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos and
                 2011-2021 Pekka Jääskeläinen
                 2023 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl-vortex.h"
#include "builtin_kernels.hh"
#include "common.h"
#include "config.h"
#include "config2.h"
#include "cpuinfo.h"
#include "devices.h"
#include "pocl_local_size.h"
#include "pocl_util.h"
#include "topology/pocl_topology.h"
#include "utlist.h"

#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <utlist.h>

#if !defined(OCS_AVAILABLE)
//#include <vortex.h>
//#include "/nethome/sjeong306/vortex-dev-16/runtime/include/vortex.h"
#include "vortex.h"
#endif

#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "common_driver.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

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

struct vx_device_data_t {
  #if !defined(OCS_AVAILABLE)
    vx_device_h vx_device;
    uint8_t* printf_buffer_ptr;
    uint32_t printf_buffer_devaddr;
    uint32_t printf_buffer_position;   
  #endif

  /* List of commands ready to be executed */
  _cl_command_node *ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* Currently loaded kernel. */
  cl_kernel current_kernel;
};

/*typedef struct _pocl_vortex_usm_allocation_t
{
  void *ptr;
  size_t size;
  cl_mem_alloc_flags_intel flags;
  unsigned alloc_type;

  struct _pocl_vortex_usm_allocation_t *next, *prev;
} pocl_vortex_usm_allocation_t;
*/

struct vx_buffer_data_t {
#if !defined(OCS_AVAILABLE)
  vx_device_h vx_device;
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

static size_t ALIGNED_CTX_SIZE = 4 * ((sizeof(struct kernel_context_t) + 3) / 4);

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

void
pocl_vortex_init_device_ops(struct pocl_device_ops *ops)
{
  ops->device_name = "vortex";

  ops->probe = pocl_vortex_probe;
  ops->uninit = pocl_vortex_uninit;
  ops->reinit = NULL;
  ops->init = pocl_vortex_init;

  ops->build_hash = pocl_vortex_build_hash;
  
  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_driver_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->compile_kernel = pocl_vortex_compile_kernel;
  ops->build_builtin = pocl_driver_build_opencl_builtins;

#if !defined(OCS_AVAILABLE)
  ops->run = pocl_vortex_run;
  ops->run_native = NULL;

  ops->alloc_mem_obj = pocl_vortex_alloc_mem_obj;
  ops->free = pocl_vortex_free;

  ops->join = pocl_vortex_join;
  ops->submit = pocl_vortex_submit;
  ops->broadcast = pocl_broadcast;
  ops->notify = pocl_vortex_notify;
  ops->flush = pocl_vortex_flush;
  ops->compute_local_size = pocl_default_local_size_optimizer;
 
  ops->read = pocl_vortex_read; // pocl_driver_read;
  ops->write = pocl_vortex_write; // pocl_driver_write;
  ops->read_rect = NULL;//pocl_driver_read_rect;  
  ops->write_rect = NULL;//pocl_driver_write_rect;
  ops->copy = NULL;//pocl_driver_copy;
  ops->copy_with_size = NULL;//pocl_driver_copy_with_size;
  ops->copy_rect = NULL;//pocl_driver_copy_rect;
  ops->memfill = NULL;//pocl_driver_memfill;
  ops->map_mem = pocl_vortex_map_mem;
  ops->unmap_mem = pocl_vortex_unmap_mem;
  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;
  ops->can_migrate_d2d = NULL;
  ops->migrate_d2d = NULL;

  ops->get_device_info_ext = NULL;
  ops->get_mem_info_ext = NULL;
  ops->set_kernel_exec_info_ext = NULL;

  ops->svm_free = NULL;
  ops->svm_alloc = NULL;
  ops->usm_alloc = NULL;
  ops->usm_free = NULL;
  ops->usm_free_blocking = NULL;
  /* no need to implement these as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_advise = NULL;
  ops->svm_migrate = NULL;
  ops->svm_copy = NULL;
  ops->svm_fill = NULL;

  ops->create_kernel = NULL;
  ops->free_kernel = NULL;
  ops->create_sampler = NULL;
  ops->free_sampler = NULL;
  ops->copy_image_rect = NULL;
  ops->write_image_rect = NULL;
  ops->read_image_rect = NULL;
  ops->map_image = NULL;
  ops->unmap_image = NULL;
  ops->fill_image = NULL;
#endif
}

char *
pocl_vortex_build_hash (cl_device_id device)
{
  char *res = (char *)calloc(1000, sizeof(char));
  //snprintf (res, 1000, "cpu-minimal-%s-%s", HOST_DEVICE_BUILD_HASH,
  //          device->llvm_cpu);
  snprintf(res, 1000, "vortex-riscv32-unknown-unknown-elf");
  return res;
}

static cl_device_partition_property vortex_partition_properties[1] = {0};

static const char *final_ld_flags[] = {"-nostartfiles", HOST_LD_FLAGS_ARRAY, NULL};

unsigned int
pocl_vortex_probe(struct pocl_device_ops *ops)
{
  if (0 == strcmp(ops->device_name, "vortex"))
    return 1;
  return 0;
}

cl_int
pocl_vortex_init (unsigned j, cl_device_id dev, const char* parameters)
{
  struct vx_device_data_t *d;
  cl_int ret = CL_SUCCESS;
  int vx_err;
  static int first_vortex_init = 1;

  if (first_vortex_init)
    {
      POCL_MSG_WARN("INIT dlcache DOTO delete\n");
      pocl_init_dlhandle_cache();
      first_vortex_init = 0;
    }
  //device->global_mem_id = 0;

  pocl_init_default_device_infos(dev);

  // TODO : temporary disable 
  //vx_err = pocl_topology_detect_device_info(dev);
  //if (vx_err != 0)
  //  return CL_INVALID_DEVICE;

  d = (struct vx_device_data_t *)calloc(1, sizeof(struct vx_device_data_t));
  if (d == NULL){
    return CL_OUT_OF_HOST_MEMORY;
  }

  // TODO : change this to vortex mem size
  dev->global_mem_size = 3*1024*1024*1024; //MIN_MAX_MEM_ALLOC_SIZE;
  dev->max_mem_alloc_size=dev->global_mem_size / 4;
  POCL_MSG_WARN("GLOBAL_MEM_SIZE  : %ld\n", dev->global_mem_size);

  dev->vendor = "Vortex Group";
  dev->vendor_id = 0;
  dev->type = CL_DEVICE_TYPE_GPU;

  dev->address_bits = 32;  
  dev->llvm_target_triplet = "riscv32";
  dev->llvm_cpu = "";

  dev->max_compute_units = 1;

  dev->long_name = "Vortex Open-Source GPU";
  dev->short_name = "Vortex";

  dev->image_support = CL_FALSE;
  dev->has_64bit_long = 0; 
  
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;
  dev->device_alloca_locals = CL_FALSE;
  
  // TODO : temporary disable 
  //pocl_cpuinfo_detect_device_info(dev);
  pocl_set_buffer_image_limits(dev); 

#if !defined(OCS_AVAILABLE)

  vx_device_h vx_device;

  vx_err = vx_dev_open(&vx_device);
  if (vx_err != 0) {
    free(d);
    return CL_DEVICE_NOT_FOUND;
  }

  uint64_t local_mem_size;
  vx_err = vx_dev_caps(vx_device, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(d);
    return CL_DEVICE_NOT_FOUND;
  }
  
  dev->device_side_printf = 1;
  dev->printf_buffer_size = PRINT_BUFFER_SIZE;
  dev->local_mem_size = local_mem_size;

  // add storage for string positions
  uint64_t printf_buffer_devaddr;
  vx_err = vx_mem_alloc(vx_device, PRINT_BUFFER_SIZE + sizeof(uint32_t), &printf_buffer_devaddr);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(d);
    return CL_INVALID_DEVICE;
  }
    
  // clear print position to zero
  uint32_t print_pos = 0;
  vx_err = vx_copy_to_dev(vx_device, printf_buffer_devaddr + PRINT_BUFFER_SIZE, &print_pos, sizeof(uint32_t));
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(d);
    return CL_OUT_OF_HOST_MEMORY;
  }

  d->printf_buffer_ptr = malloc(PRINT_BUFFER_SIZE);
  d->printf_buffer_devaddr = printf_buffer_devaddr;
  d->printf_buffer_position = printf_buffer_devaddr + PRINT_BUFFER_SIZE;    
  d->vx_device = vx_device;

#endif 

  d->current_kernel = NULL;

  POCL_INIT_LOCK(d->cq_lock);
  
  dev->data = d;  

  //SETUP_DEVICE_CL_VERSION(1, 2);
#if (HOST_DEVICE_CL_VERSION_MAJOR >= 3)
  // TODO : opencl_3d_images, opencl_c_fp64 has some redefine issue, https://reviews.llvm.org/D106260.
  dev->features = "__opencl_c_images";
  // dev->features = "__opencl_c_3d_image_writes  __opencl_c_images   __opencl_c_atomic_order_acq_rel __opencl_c_atomic_order_seq_cst   __opencl_c_atomic_scope_device __opencl_c_program_scope_global_  variables   __opencl_c_generic_address_space __opencl_c_subgroups __opencl_c_atomic_scope_all_devices __opencl_c_read_write_images __opencl_c_fp16 __opencl_c_fp64 __opencl_c_int64";
  //dev->features = " __opencl_c_images   __opencl_c_atomic_order_acq_rel __opencl_c_atomic_order_seq_cst   __opencl_c_atomic_scope_device __opencl_c_program_scope_global_  variables   __opencl_c_generic_address_space __opencl_c_subgroups __opencl_c_atomic_scope_all_devices __opencl_c_read_write_images __opencl_c_fp16  __opencl_c_int64";

  dev->program_scope_variables_pass = CL_TRUE;
  dev->generic_as_support = CL_TRUE;

  pocl_setup_opencl_c_with_version (dev, CL_TRUE);
  pocl_setup_features_with_version (dev);
#else
  pocl_setup_opencl_c_with_version (dev, CL_FALSE);
#endif

  //pocl_setup_extensions_with_version (dev);
  char extensions[1024];
  extensions[0] = 0;
  strcat (extensions, "cl_khr_byte_addressable_store"
                      " cl_khr_global_int32_base_atomics"
                      " cl_khr_global_int32_extended_atomics"
                      " cl_khr_local_int32_base_atomics"
                      " cl_khr_local_int32_extended_atomics"); 
  /*strcat (extensions, //" cl_khr_3d_image_writes"
                      //" cl_khr_spir"
                      //" cl_khr_fp16"
                      //" cl_khr_int64_base_atomics"
                      //" cl_khr_int64_extended_atomics"
                      //" cl_khr_fp64"
                      ); */
  dev->extensions = strdup (extensions);

  return ret;
}

cl_int pocl_vortex_uninit (unsigned j, cl_device_id device) {
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;
  if (NULL == d)
    return CL_SUCCESS;  

#if !defined(OCS_AVAILABLE)
  free(d->printf_buffer_ptr);
  vx_mem_free(d->vx_device, d->printf_buffer_devaddr);
  vx_dev_close(d->vx_device);
#endif

  POCL_DESTROY_LOCK (d->cq_lock);
  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}


#if !defined(OCS_AVAILABLE)

cl_int pocl_vortex_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void *host_ptr) {
  int vx_err;
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];

  /* let other drivers preallocate */
  if ((mem_obj->flags & CL_MEM_ALLOC_HOST_PTR) && (mem_obj->mem_host_ptr == NULL))
    return CL_MEM_OBJECT_ALLOCATION_FAILURE;

  p->extra_ptr = NULL;
  p->version = 0;
  p->extra = 0;
  cl_mem_flags flags = mem_obj->flags;
  
  if (flags & CL_MEM_USE_HOST_PTR) {
    POCL_ABORT("POCL_VORTEX_MALLOC\n");
  } else {
    struct vx_device_data_t* d = (struct vx_device_data_t *)device->data;

    size_t dev_mem_addr;
    vx_err = vx_mem_alloc(d->vx_device, mem_obj->size, &dev_mem_addr);
    if (vx_err != 0) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    if (host_ptr && (flags & CL_MEM_COPY_HOST_PTR)) {
      vx_err = vx_copy_to_dev(d->vx_device, dev_mem_addr, host_ptr, mem_obj->size);
      if (vx_err != 0) {
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
    }

    if (flags & CL_MEM_ALLOC_HOST_PTR) {
      /* malloc mem_host_ptr then increase refcount */
      pocl_alloc_or_retain_mem_host_ptr (mem_obj);
    }

    struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t *)malloc(sizeof(struct vx_buffer_data_t));
    buf_data->vx_device = d->vx_device;
    buf_data->dev_mem_addr = dev_mem_addr;
    
    p->mem_ptr = buf_data;
  }

  return CL_SUCCESS;
}

void pocl_vortex_free(cl_device_id device, cl_mem mem_obj) {
  pocl_mem_identifier *p = &mem_obj->device_ptrs[device->global_mem_id];
  cl_mem_flags flags = mem_obj->flags;
  struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t*)p->mem_ptr;

  if (flags & CL_MEM_USE_HOST_PTR) {
    POCL_ABORT("POCL_VORTEX_FREE\n"); 
  } else {    
    if (flags & CL_MEM_ALLOC_HOST_PTR) {
      pocl_release_mem_host_ptr(mem_obj);
    }
    vx_mem_free(buf_data->vx_device, buf_data->dev_mem_addr);
  }  
  free(buf_data);
  p->mem_ptr = NULL;
  p->version = 0;
}

void pocl_vortex_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset, 
                       size_t size) {
  int vx_err;
  struct vx_buffer_data_t *buf_data = (struct vx_buffer_data_t *)dst_mem_id->mem_ptr;
  vx_err = vx_copy_to_dev(buf_data->vx_device, buf_data->dev_mem_addr + offset, host_ptr, size);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_WRITE\n");
  }
}

void pocl_vortex_read(void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t offset, 
                      size_t size) {
  int vx_err;                 
  struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t*)src_mem_id->mem_ptr;
  vx_err = vx_copy_from_dev(buf_data->vx_device, host_ptr, buf_data->dev_mem_addr + offset, size);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_READ\n");
  }
}

cl_int pocl_vortex_map_mem (void *data, pocl_mem_identifier *src_mem_id,
                            cl_mem src_buf, mem_mapping_t *map) {
  if (map->map_flags & CL_MAP_WRITE_INVALIDATE_REGION)
    return CL_SUCCESS;

  int vx_err;
  struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t*)src_mem_id->mem_ptr;
  vx_err = vx_copy_from_dev(buf_data->vx_device, map->host_ptr, buf_data->dev_mem_addr + map->offset, map->size);
  if (vx_err != 0) {
    return CL_MAP_FAILURE;
  }

  return CL_SUCCESS;
}

cl_int pocl_vortex_unmap_mem (void *data, pocl_mem_identifier *dst_mem_id,
                              cl_mem dst_buf, mem_mapping_t *map) {
  if (map->map_flags == CL_MAP_READ)
    return CL_SUCCESS;

  int vx_err;
  struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t*)dst_mem_id->mem_ptr;
  vx_err = vx_copy_to_dev(buf_data->vx_device, buf_data->dev_mem_addr + map->offset, map->host_ptr, map->size);
  if (vx_err != 0) {
    return CL_MAP_FAILURE;
  }

  return CL_SUCCESS;
}

static void vortex_command_scheduler (struct vx_device_data_t *d) 
{
  _cl_command_node *node;
  
  /* execute commands from ready list */
  while ((node = d->ready_list))
    {
      assert (pocl_command_is_ready (node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (d->ready_list, node);
      POCL_UNLOCK (d->cq_lock);
      pocl_exec_command (node);
      POCL_LOCK (d->cq_lock);
    }

  return;
}

void
pocl_vortex_submit (_cl_command_node *node, cl_command_queue cq)
{
  struct vx_device_data_t *d = (struct vx_device_data_t *)node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (node, CL_TRUE, CL_TRUE);

  node->ready = 1;
  POCL_LOCK (d->cq_lock);
  pocl_command_push(node, &d->ready_list, &d->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  vortex_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void pocl_vortex_flush (cl_device_id device, cl_command_queue cq)
{
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  vortex_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);
}

void
pocl_vortex_join (cl_device_id device, cl_command_queue cq)
{
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;

  POCL_LOCK (d->cq_lock);
  vortex_command_scheduler (d);
  POCL_UNLOCK (d->cq_lock);

  return;
}

void
pocl_vortex_notify (cl_device_id device, cl_event event, cl_event finished)
{
  struct vx_device_data_t *d = (struct vx_device_data_t *)device->data;
  _cl_command_node * volatile node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (!node->ready)
    return;

  if (pocl_command_is_ready (event))
    {
      if (event->status == CL_QUEUED)
        {
          pocl_update_event_submitted (event);
          POCL_LOCK (d->cq_lock);
          CDL_DELETE (d->command_list, node);
          CDL_PREPEND (d->ready_list, node);
          vortex_command_scheduler (d);
          POCL_UNLOCK (d->cq_lock);
        }
      return;
    }
}

void
pocl_vortex_run (void *data, _cl_command_node *cmd)
{
  struct vx_device_data_t *d;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;
  cl_uint dev_i = cmd->program_device_i;
  uint64_t local_mem_base_addr;
  int vx_err;

  pocl_driver_build_gvar_init_kernel(
    program, dev_i, cmd->device, pocl_cpu_gvar_init_callback
  );

  if (pc->num_groups[0] == 0 
   || pc->num_groups[1] == 0 
   || pc->num_groups[2] == 0)
    return;

  assert (data != NULL);
  d = (struct vx_device_data_t *)data;

  // calculate kernel arguments buffer size
  size_t abuf_args_size = 4 * (meta->num_args + meta->num_locals);
  size_t abuf_size = ALIGNED_CTX_SIZE + abuf_args_size; 
  size_t local_mem_size = 0;
  
  for (i = 0; i < meta->num_args; ++i) {  
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);  
    if (ARG_IS_LOCAL(meta->arg_info[i])) {        
      abuf_size += 4;
      local_mem_size += al->size;
    } else
    if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
      || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
      || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
      abuf_size += 4;
    } else {
      // scalar argument
      abuf_size += al->size;
    }
  }

  // non-argument local buffers
  for (i = 0; i < meta->num_locals; ++i) {
    local_mem_size += meta->local_sizes[i];
    abuf_size += 4;
  }

  // allocate local buffer
  if (local_mem_size) {
    uint64_t dev_local_mem_size;
    vx_err = vx_dev_caps(d->vx_device, VX_CAPS_LOCAL_MEM_SIZE, &dev_local_mem_size);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    if (local_mem_size > dev_local_mem_size) {
      POCL_MSG_ERR("Local memory allocation failed: out of memory needed=%ld, available=%ld\n", local_mem_size, dev_local_mem_size);
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    vx_err = vx_dev_caps(d->vx_device, VX_CAPS_LOCAL_MEM_ADDR, &local_mem_base_addr);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }
  }

  // allocate arguments buffer
  uint64_t args_dev_base_addr = KERNEL_ARG_BASE_ADDR;
  uint8_t* args_host_base_ptr = malloc(abuf_size);
  assert(args_host_base_ptr);

  // write context data
  {
    struct kernel_context_t ctx;
    for (int i = 0; i < 3; ++i) {
      ctx.num_groups[i] = pc->num_groups[i];
      ctx.global_offset[i] = pc->global_offset[i];
      ctx.local_size[i] = pc->local_size[i];        
    }
    ctx.work_dim = pc->work_dim;      
    ctx.printf_buffer = d->printf_buffer_devaddr;
    ctx.printf_buffer_position = d->printf_buffer_position;
    ctx.printf_buffer_capacity = PRINT_BUFFER_SIZE;

    memset(args_host_base_ptr, 0, ALIGNED_CTX_SIZE);
    memcpy(args_host_base_ptr, &ctx, sizeof(struct kernel_context_t));
  }

  // write arguments
  
  uint8_t* args_host_ptr  = args_host_base_ptr + ALIGNED_CTX_SIZE;
  uint32_t data_dev_addr = args_dev_base_addr + ALIGNED_CTX_SIZE + abuf_args_size;
  uint64_t local_mem_addr = local_mem_base_addr;

  for (i = 0; i < meta->num_args; ++i) {
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      memcpy(args_host_ptr, &data_dev_addr, 4); // pointer index
      memcpy(args_host_base_ptr + (data_dev_addr - args_dev_base_addr), &local_mem_addr, 4); // pointer value          
      local_mem_addr += al->size;
      args_host_ptr += 4;
      data_dev_addr += 4;
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      memcpy(args_host_ptr, &data_dev_addr, 4); // pointer index     
      if (al->value == NULL) {
        memset(args_host_base_ptr + (data_dev_addr - args_dev_base_addr), 0, 4); // NULL pointer value 
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        //auto buf_data = (vx_buffer_data_t*)m->device_ptrs[cmd->device->dev_id].mem_ptr;
        struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t *) m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
        void* dev_mem_addr = (void*)(buf_data->dev_mem_addr + al->offset);
        memcpy(args_host_base_ptr + (data_dev_addr - args_dev_base_addr), &dev_mem_addr, 4); // pointer value
      }
      args_host_ptr += 4;
      data_dev_addr += 4;
    } else 
    if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
        POCL_ABORT("POCL_VORTEX_RUN\n"); 
    } else 
    if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
        POCL_ABORT("POCL_VORTEX_RUN\n"); 
    } else {
      // scalar argument
      memcpy(args_host_ptr, &data_dev_addr, 4); // scalar index
      memcpy(args_host_base_ptr + (data_dev_addr - args_dev_base_addr), al->value, al->size); // scalar value
      args_host_ptr += 4;
      data_dev_addr += al->size;
    }
  }

  // write local arguments
  for (i = 0; i < meta->num_locals; ++i) {
    memcpy(args_host_ptr, &data_dev_addr, 4); // pointer index
    memcpy(args_host_base_ptr + (data_dev_addr - args_dev_base_addr), &local_mem_addr, 4); // pointer value          
    local_mem_addr += meta->local_sizes[i];
    args_host_ptr += 4;
    data_dev_addr += 4;
  }

  // upload kernel arguments buffer
  vx_err = vx_copy_to_dev(d->vx_device, args_dev_base_addr, args_host_base_ptr, abuf_size);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // release staging buffer
  free(args_host_base_ptr);
  
  // upload kernel to device
  if (NULL == d->current_kernel 
    || d->current_kernel != kernel) {    
      d->current_kernel = kernel;
    char program_bin_path[POCL_MAX_FILENAME_LENGTH];
    pocl_cache_final_binary_path (program_bin_path, program, dev_i, kernel, NULL, 0);
    vx_err = vx_upload_kernel_file(d->vx_device, program_bin_path);      
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }
  }

    
  // quick off kernel execution
  vx_err = vx_start(d->vx_device);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // wait for the execution to complete
  vx_err = vx_ready_wait(d->vx_device, -1);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  vx_dump_perf(d->vx_device, stdout);
  
  {
    // flush print buffer 
    uint32_t print_size;
    vx_err = vx_copy_from_dev(d->vx_device, &print_size, d->printf_buffer_devaddr + PRINT_BUFFER_SIZE, sizeof(uint32_t));
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    if (print_size != 0) {
      // read print buffer data
      vx_err = vx_copy_from_dev(d->vx_device, d->printf_buffer_ptr, d->printf_buffer_devaddr, print_size);
      if (vx_err != 0) {
        POCL_ABORT("POCL_VORTEX_RUN\n");
      }    
      
      // write print buffer to console stdout
      write(STDOUT_FILENO, d->printf_buffer_ptr, print_size);
      
      // reset device print buffer
      print_size = 0;     
      vx_err = vx_copy_to_dev(d->vx_device, d->printf_buffer_devaddr + PRINT_BUFFER_SIZE, &print_size, sizeof(uint32_t));
      if (vx_err != 0) {
        POCL_ABORT("POCL_VORTEX_RUN\n");
      }
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

/*
#define WORKGROUP_STRING_LENGTH 1024
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
  char kernel_elf[POCL_MAX_FILENAME_LENGTH];
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

  std::string clang_path(CLANG);
  if (llvm_install_path) {
    clang_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
  }

  {
    std::stringstream ss_cmd, ss_out;
    ss_cmd << clang_path.c_str() << " " << build_cflags << " " << kernel_bc << " -c -o " << kernel_obj;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }
  
  {  
    char wrapper_cc[POCL_MAX_FILENAME_LENGTH];    
    char pfn_workgroup_string[WORKGROUP_STRING_LENGTH];
    std::stringstream ss;

    snprintf (pfn_workgroup_string, WORKGROUP_STRING_LENGTH,
              "_pocl_kernel_%s_workgroup", kernel->name);
    
    ss << "#include <vx_spawn.h>\n"
          "void " << pfn_workgroup_string << "(uint8_t* args, uint8_t* ctx, uint32_t group_x, uint32_t group_y, uint32_t group_z);\n"  
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
    ss_cmd << objdump_path.c_str() << " -D " << kernel_elf << " > " << kernel->name << ".dump";
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }
  
  return 0;
}
*/
