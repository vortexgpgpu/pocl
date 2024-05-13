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
#include <vortex.h>
#endif

#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "common_driver.h"
#include "pocl-vortex-config.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

/* Maximum kernels occupancy */
#define MAX_KERNELS 16

/* default WG size in each dimension & total WG size.
 * this should be reasonable for CPU */
#define DEFAULT_WG_SIZE 4096

// allocate 1MB OpenCL print buffer
#define PRINT_BUFFER_SIZE (1024 * 1024)

extern char *build_cflags;
extern char *build_ldflags;

struct vx_device_data_t {
  #if !defined(OCS_AVAILABLE)
    vx_device_h vx_device;
    vx_buffer_h vx_kernel_buffer;
  #endif

  /* List of commands ready to be executed */
  _cl_command_node *ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* Currently loaded kernel. */
  cl_kernel current_kernel;

  int is64bit;
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
  vx_buffer_h vx_buffer;
#endif
};

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
  struct vx_device_data_t *dd = (struct vx_device_data_t *)device->data;
  if (dd->is64bit) {
    snprintf(res, 1000, "vortex-riscv64-unknown-unknown-elf");
  } else {
    snprintf(res, 1000, "vortex-riscv32-unknown-unknown-elf");
  }
  return res;
}

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
  struct vx_device_data_t *dd;
  cl_int ret = CL_SUCCESS;
  int vx_err;
  static int first_vortex_init = 1;

  const char* sz_cflags = pocl_get_string_option("POCL_VORTEX_CFLAGS", "");

  int is64bit = (strstr(sz_cflags, "-march=rv64") != NULL) ? 1 : 0;

  if (first_vortex_init) {
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

  dd = (struct vx_device_data_t *)calloc(1, sizeof(struct vx_device_data_t));
  if (dd == NULL){
    return CL_OUT_OF_HOST_MEMORY;
  }

  dev->vendor = "Vortex Group";
  dev->vendor_id = 0;
  dev->type = CL_DEVICE_TYPE_GPU;

  dev->address_bits = is64bit ? 64 : 32;
  dev->llvm_target_triplet = is64bit ? "riscv64" : "riscv32";
  dev->llvm_cpu = "";

  dev->global_mem_size = 3*1024*1024*1024;
  dev->max_compute_units = 1;

  dev->device_side_printf = 0;

  dev->long_name = "Vortex Open-Source GPU";
  dev->short_name = "Vortex";

  dev->image_support = CL_FALSE;
  dev->has_64bit_long = is64bit;

  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;
  dev->device_alloca_locals = CL_FALSE;

  // TODO : temporary disable
  //pocl_cpuinfo_detect_device_info(dev);
  pocl_set_buffer_image_limits(dev);

#if !defined(OCS_AVAILABLE)

  vx_device_h vx_device;

  vx_err = vx_dev_open(&vx_device);
  if (vx_err != 0) {
    free(dd);
    return CL_DEVICE_NOT_FOUND;
  }

  uint64_t num_cores;
  vx_err = vx_dev_caps(vx_device, VX_CAPS_NUM_CORES, &num_cores);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(dd);
    return CL_DEVICE_NOT_FOUND;
  }

  uint64_t global_mem_size;
  vx_err = vx_dev_caps(vx_device, VX_CAPS_GLOBAL_MEM_SIZE, &global_mem_size);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(dd);
    return CL_DEVICE_NOT_FOUND;
  }

  uint64_t local_mem_size;
  vx_err = vx_dev_caps(vx_device, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(dd);
    return CL_DEVICE_NOT_FOUND;
  }

  dev->printf_buffer_size = PRINT_BUFFER_SIZE;
  dev->global_mem_size = global_mem_size;
  dev->local_mem_size = local_mem_size;
  dev->max_compute_units = num_cores;

  // add storage for string positions
  vx_buffer_h vx_printf_buffer;
  vx_err = vx_mem_alloc(vx_device, PRINT_BUFFER_SIZE + sizeof(uint32_t), VX_MEM_READ_WRITE, &vx_printf_buffer);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(dd);
    return CL_INVALID_DEVICE;
  }

  // clear print position to zero
  uint32_t print_pos = 0;
  vx_err = vx_copy_to_dev(vx_printf_buffer, &print_pos, PRINT_BUFFER_SIZE, sizeof(uint32_t));
  if (vx_err != 0) {
    vx_mem_free(vx_printf_buffer);
    vx_dev_close(vx_device);
    free(dd);
    return CL_OUT_OF_HOST_MEMORY;
  }

  dd->vx_kernel_buffer = NULL;
  dd->vx_device = vx_device;

#endif

  dd->current_kernel = NULL;
  dd->is64bit = is64bit;

  POCL_INIT_LOCK(dd->cq_lock);

  dev->data = dd;

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
  strcat (extensions, "cl_khr_byte_addressable_store");
  dev->extensions = strdup (extensions);

  return ret;
}

cl_int pocl_vortex_uninit (unsigned j, cl_device_id device) {
  struct vx_device_data_t *dd = (struct vx_device_data_t *)device->data;
  if (NULL == dd)
    return CL_SUCCESS;

#if !defined(OCS_AVAILABLE)
  if (dd->vx_kernel_buffer != NULL) {
    vx_mem_free(dd->vx_kernel_buffer);
  }
  vx_dev_close(dd->vx_device);
#endif

  POCL_DESTROY_LOCK (dd->cq_lock);
  POCL_MEM_FREE(dd);
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
    int vx_flags = 0;
    if ((flags & CL_MEM_READ_WRITE) != 0)
      vx_flags = VX_MEM_READ_WRITE;
    if ((flags & CL_MEM_READ_ONLY) != 0)
      vx_flags = VX_MEM_READ;
    if ((flags & CL_MEM_WRITE_ONLY) != 0)
      vx_flags = VX_MEM_WRITE;

    struct vx_device_data_t* dd = (struct vx_device_data_t *)device->data;

    vx_buffer_h vx_buffer;
    vx_err = vx_mem_alloc(dd->vx_device, mem_obj->size, vx_flags, &vx_buffer);
    if (vx_err != 0) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    if (host_ptr && (flags & CL_MEM_COPY_HOST_PTR)) {
      vx_err = vx_copy_to_dev(vx_buffer, host_ptr, 0, mem_obj->size);
      if (vx_err != 0) {
        return CL_MEM_OBJECT_ALLOCATION_FAILURE;
      }
    }

    if (flags & CL_MEM_ALLOC_HOST_PTR) {
      /* malloc mem_host_ptr then increase refcount */
      pocl_alloc_or_retain_mem_host_ptr (mem_obj);
    }

    struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t *)malloc(sizeof(struct vx_buffer_data_t));
    buf_data->vx_device = dd->vx_device;
    buf_data->vx_buffer = vx_buffer;

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
    vx_mem_free(buf_data->vx_buffer);
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
  vx_err = vx_copy_to_dev(buf_data->vx_buffer, host_ptr, offset, size);
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
  vx_err = vx_copy_from_dev(host_ptr, buf_data->vx_buffer, offset, size);
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
  vx_err = vx_copy_from_dev(map->host_ptr, buf_data->vx_buffer, map->offset, map->size);
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
  vx_err = vx_copy_to_dev(buf_data->vx_buffer, map->host_ptr, map->offset, map->size);
  if (vx_err != 0) {
    return CL_MAP_FAILURE;
  }

  return CL_SUCCESS;
}

static void vortex_command_scheduler (struct vx_device_data_t *dd)
{
  _cl_command_node *node;

  /* execute commands from ready list */
  while ((node = dd->ready_list))
    {
      assert (pocl_command_is_ready (node->sync.event.event));
      assert (node->sync.event.event->status == CL_SUBMITTED);
      CDL_DELETE (dd->ready_list, node);
      POCL_UNLOCK (dd->cq_lock);
      pocl_exec_command (node);
      POCL_LOCK (dd->cq_lock);
    }

  return;
}

void
pocl_vortex_submit (_cl_command_node *node, cl_command_queue cq)
{
  struct vx_device_data_t *dd = (struct vx_device_data_t *)node->device->data;

  if (node != NULL && node->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache (node, CL_TRUE, CL_TRUE);

  node->ready = 1;
  POCL_LOCK (dd->cq_lock);
  pocl_command_push(node, &dd->ready_list, &dd->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  vortex_command_scheduler (dd);
  POCL_UNLOCK (dd->cq_lock);

  return;
}

void pocl_vortex_flush (cl_device_id device, cl_command_queue cq)
{
  struct vx_device_data_t *dd = (struct vx_device_data_t *)device->data;

  POCL_LOCK (dd->cq_lock);
  vortex_command_scheduler (dd);
  POCL_UNLOCK (dd->cq_lock);
}

void
pocl_vortex_join (cl_device_id device, cl_command_queue cq)
{
  struct vx_device_data_t *dd = (struct vx_device_data_t *)device->data;

  POCL_LOCK (dd->cq_lock);
  vortex_command_scheduler (dd);
  POCL_UNLOCK (dd->cq_lock);

  return;
}

void
pocl_vortex_notify (cl_device_id device, cl_event event, cl_event finished)
{
  struct vx_device_data_t *dd = (struct vx_device_data_t *)device->data;
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
          POCL_LOCK (dd->cq_lock);
          CDL_DELETE (dd->command_list, node);
          CDL_PREPEND (dd->ready_list, node);
          vortex_command_scheduler (dd);
          POCL_UNLOCK (dd->cq_lock);
        }
      return;
    }
}

void
pocl_vortex_run (void *data, _cl_command_node *cmd)
{
  struct vx_device_data_t *dd;
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

  int num_groups = 1;
  for (int i = 0; i < pc->work_dim; ++i) {
    num_groups *= pc->num_groups[i];
  }
  if (num_groups == 0)
    return;

  assert (data != NULL);
  dd = (struct vx_device_data_t *)data;

  int ptr_size = dd->is64bit ? 8 : 4;

  // calculate kernel arguments buffer size
  size_t abuf_args_size = ptr_size * (meta->num_args + meta->num_locals);
  size_t abuf_size = ALIGNED_CTX_SIZE + abuf_args_size;
  int local_mem_size = 0;

  for (i = 0; i < meta->num_args; ++i) {
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      abuf_size += ptr_size;
      local_mem_size += al->size;
    } else
    if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
     || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
     || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
      abuf_size += ptr_size;
    } else {
      // scalar argument
      abuf_size += al->size;
    }
  }

  // local buffers
  for (i = 0; i < meta->num_locals; ++i) {
    local_mem_size += meta->local_sizes[i];
    abuf_size += ptr_size;
  }

  // allocate local buffer
  int occupancy = 0;
  if (local_mem_size != 0) {
    uint64_t dev_local_mem_size;
    vx_err = vx_dev_caps(dd->vx_device, VX_CAPS_LOCAL_MEM_SIZE, &dev_local_mem_size);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    uint64_t num_cores;
    vx_err = vx_dev_caps(dd->vx_device, VX_CAPS_NUM_CORES, &num_cores);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    uint64_t warps_per_core;
    vx_err = vx_dev_caps(dd->vx_device, VX_CAPS_NUM_WARPS, &warps_per_core);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    uint64_t threads_per_warp;
    vx_err = vx_dev_caps(dd->vx_device, VX_CAPS_NUM_THREADS, &threads_per_warp);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }

    // calculate core occupancy
    int threads_per_core = warps_per_core * threads_per_warp;
    int needed_cores = (num_groups + threads_per_core - 1) / threads_per_core;
    int active_cores = min(needed_cores, num_cores);
    int total_wgs_per_core = (num_groups + active_cores - 1) / active_cores;
    occupancy = min(total_wgs_per_core, threads_per_core);

    // check if local memory is available
    uint64_t local_memory_working_set = occupancy * local_mem_size;
    if (local_memory_working_set > dev_local_mem_size) {
      POCL_ABORT("out of local memory: needed=%ld bytes (%d active workgoups x %d bytes), available=%ld bytes\n",
        local_memory_working_set, occupancy, local_mem_size, dev_local_mem_size);
    }

    vx_err = vx_dev_caps(dd->vx_device, VX_CAPS_LOCAL_MEM_ADDR, &local_mem_base_addr);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }
  }

  // allocate arguments host buffer
  uint8_t* const host_args_base_ptr = malloc(abuf_size);
  assert(host_args_base_ptr);

  // allocate arguments device buffer
  vx_buffer_h vx_args_buffer;
  vx_err = vx_mem_alloc(dd->vx_device, abuf_size, VX_MEM_READ, &vx_args_buffer);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  uint64_t dev_args_base_addr;
  vx_err = vx_mem_address(vx_args_buffer, &dev_args_base_addr);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // write context data
  {
    pocl_kernel_context_t* ctx = (pocl_kernel_context_t*)host_args_base_ptr;
    for (int i = 0; i < 3; ++i) {
      ctx->num_groups[i] = pc->num_groups[i];
      ctx->global_offset[i] = pc->global_offset[i];
      ctx->local_size[i] = pc->local_size[i];
    }
    ctx->work_dim = pc->work_dim;
  }

  // write arguments

  uint8_t* host_args_ptr = host_args_base_ptr + ALIGNED_CTX_SIZE;
  uint32_t dev_data_addr = dev_args_base_addr + ALIGNED_CTX_SIZE + abuf_args_size;
  uint64_t local_mem_addr = local_mem_base_addr;

  for (i = 0; i < meta->num_args; ++i) {
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      memcpy(host_args_ptr, &dev_data_addr, ptr_size); // pointer index
      memcpy(host_args_base_ptr + (dev_data_addr - dev_args_base_addr), &local_mem_addr, ptr_size); // pointer value
      local_mem_addr += occupancy * al->size;
      host_args_ptr += ptr_size;
      dev_data_addr += ptr_size;
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      memcpy(host_args_ptr, &dev_data_addr, ptr_size); // pointer index
      if (al->value == NULL) {
        memset(host_args_base_ptr + (dev_data_addr - dev_args_base_addr), 0, ptr_size); // NULL pointer value
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        //auto buf_data = (vx_buffer_data_t*)m->device_ptrs[cmd->device->dev_id].mem_ptr;
        struct vx_buffer_data_t* buf_data = (struct vx_buffer_data_t *) m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
        uint64_t buf_address;
        vx_err = vx_mem_address(buf_data->vx_buffer, &buf_address);
        if (vx_err != 0) {
          POCL_ABORT("POCL_VORTEX_RUN\n");
        }
        uint64_t dev_mem_addr = buf_address + al->offset;
        memcpy(host_args_base_ptr + (dev_data_addr - dev_args_base_addr), &dev_mem_addr, ptr_size); // pointer value
      }
      host_args_ptr += ptr_size;
      dev_data_addr += ptr_size;
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
        POCL_ABORT("POCL_VORTEX_RUN\n");
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
        POCL_ABORT("POCL_VORTEX_RUN\n");
    } else {
      // scalar argument
      memcpy(host_args_ptr, &dev_data_addr, ptr_size); // scalar index
      memcpy(host_args_base_ptr + (dev_data_addr - dev_args_base_addr), al->value, al->size); // scalar value
      host_args_ptr += ptr_size;
      dev_data_addr += al->size;
    }
  }

  // write local arguments
  for (i = 0; i < meta->num_locals; ++i) {
    memcpy(host_args_ptr, &dev_data_addr, ptr_size); // pointer index
    memcpy(host_args_base_ptr + (dev_data_addr - dev_args_base_addr), &local_mem_addr, ptr_size); // pointer value
    local_mem_addr += occupancy * meta->local_sizes[i];
    host_args_ptr += ptr_size;
    dev_data_addr += ptr_size;
  }

  // upload kernel arguments buffer
  vx_err = vx_copy_to_dev(vx_args_buffer, host_args_base_ptr, 0, abuf_size);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // release argument host buffer
  free(host_args_base_ptr);

  // upload kernel to device
  if (NULL == dd->current_kernel
   || dd->current_kernel != kernel) {
      dd->current_kernel = kernel;

    if (dd->vx_kernel_buffer != NULL) {
      vx_mem_free(dd->vx_kernel_buffer);
    }

    char program_bin_path[POCL_MAX_FILENAME_LENGTH];
    pocl_cache_final_binary_path (program_bin_path, program, dev_i, kernel, NULL, 0);

    vx_err = vx_upload_kernel_file(dd->vx_device, program_bin_path, &dd->vx_kernel_buffer);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }
  }

  // launch kernel execution
  vx_err = vx_start(dd->vx_device, dd->vx_kernel_buffer, vx_args_buffer);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // wait for the execution to complete
  vx_err = vx_ready_wait(dd->vx_device, -1);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // release arguments device buffer
  vx_mem_free(vx_args_buffer);

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
