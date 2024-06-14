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

#include "pocl_context.h"
#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_mem_management.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#include "common_driver.h"
#include "pocl_llvm.h"

#include "vortex_utils.h"
#include "kernel_args.h"
#include <vortex.h>

typedef struct {
  vx_device_h vx_device;
  vx_buffer_h vx_kernel_buffer;

  /* List of commands ready to be executed */
  _cl_command_node *ready_list;

  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;

  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  pocl_lock_t compile_lock;

  int is_64bit;

  size_t ctx_refcount;
} vortex_device_data_t;

typedef struct {
  int num_kernels;
  char* kernel_names;
} vortex_program_data_t;

typedef struct {
  size_t refcount;
  int kernel_id;
} vortex_kernel_data_t;

typedef struct {
  vx_device_h vx_device;
  vx_buffer_h vx_buffer;
  uint64_t buf_address;
} vortex_buffer_data_t;

static cl_bool vortex_available = CL_TRUE;

static const char *vortex_native_device_aux_funcs[] = {NULL};

char* pocl_vortex_init_build(void *data) {
    return strdup("-target-feature +m -target-feature +a -target-feature +f -target-feature +d");
}

void pocl_vortex_init_device_ops(struct pocl_device_ops *ops) {

  ops->device_name = "vortex";
  ops->build_hash = pocl_vortex_build_hash;
  ops->probe = pocl_vortex_probe;
  ops->uninit = pocl_vortex_uninit;
  ops->init = pocl_vortex_init;

  ops->init_context = pocl_vortex_init_context;
  ops->free_context = pocl_vortex_free_context;

  ops->run = pocl_vortex_run;
  ops->run_native = NULL;

  ops->alloc_mem_obj = pocl_vortex_alloc_mem_obj;
  ops->free = pocl_vortex_free;

  ops->build_source = pocl_driver_build_source;
  ops->link_program = pocl_driver_link_program;
  ops->build_binary = pocl_driver_build_binary;
  ops->free_program = pocl_driver_free_program;
  ops->setup_metadata = pocl_driver_setup_metadata;
  ops->supports_binary = pocl_driver_supports_binary;
  ops->build_poclbinary = pocl_driver_build_poclbinary;
  ops->build_builtin = pocl_driver_build_opencl_builtins;
  ops->init_build = pocl_vortex_init_build;

  ops->post_build_program = pocl_vortex_post_build_program;
  ops->free_program = pocl_vortex_free_program;

  ops->create_kernel = pocl_vortex_create_kernel;
  ops->free_kernel = pocl_vortex_free_kernel;

  ops->submit = pocl_vortex_submit;
  ops->join = pocl_vortex_join;
  ops->flush = pocl_vortex_flush;
  ops->notify = pocl_vortex_notify;
  ops->broadcast = pocl_broadcast;

  ops->read = pocl_vortex_read;
  ops->write = pocl_vortex_write;

  ops->get_mapping_ptr = pocl_driver_get_mapping_ptr;
  ops->free_mapping_ptr = pocl_driver_free_mapping_ptr;
}

char * pocl_vortex_build_hash (cl_device_id dev)
{
  char *res = (char *)calloc(1000, sizeof(char));
  if (dev->address_bits == 64) {
    snprintf(res, 1000, "vortex-riscv64-unknown-unknown-elf");
  } else {
    snprintf(res, 1000, "vortex-riscv32-unknown-unknown-elf");
  }
  return res;
}

unsigned int pocl_vortex_probe(struct pocl_device_ops *ops)
{
  return (0 == strcmp(ops->device_name, "vortex"));
}

cl_int
pocl_vortex_init (unsigned j, cl_device_id dev, const char* parameters)
{
  int vx_err;
  vortex_device_data_t *dd;

  const char* sz_xlen = pocl_get_string_option("POCL_VORTEX_XLEN", "32");

  int is_64bit = (strcmp(sz_xlen, "64") == 0);

  assert (dev->data == NULL);

  pocl_init_default_device_infos(dev, VORTEX_DEVICE_EXTENSIONS);

  SETUP_DEVICE_CL_VERSION (dev, VORTEX_DEVICE_CL_VERSION_MAJOR,
                           VORTEX_DEVICE_CL_VERSION_MINOR);

  dd = (vortex_device_data_t *)calloc(1, sizeof(vortex_device_data_t));
  if (dd == NULL){
    return CL_OUT_OF_HOST_MEMORY;
  }

  dev->vendor = "Vortex Group";
  dev->long_name = "Vortex OpenGPU";
  dev->short_name = "Vortex";
  dev->vendor_id = 0;
  dev->type = CL_DEVICE_TYPE_GPU;

  dev->spmd = CL_TRUE;
  dev->run_workgroup_pass = CL_FALSE;
  dev->execution_capabilities = CL_EXEC_KERNEL;
  //dev->global_as_id = VX_ADDR_SPACE_GLOBAL;
  //dev->local_as_id = VX_ADDR_SPACE_LOCAL;439
  //dev->constant_as_id = VX_ADDR_SPACE_CONSTANT;
  dev->autolocals_to_args = POCL_AUTOLOCALS_TO_ARGS_ALWAYS;
  dev->device_alloca_locals = CL_FALSE;
  dev->device_side_printf = 0;
  dev->has_64bit_long = is_64bit;

  dev->llvm_cpu = NULL;
  dev->address_bits = is_64bit ? 64 : 32;
  dev->llvm_target_triplet = is_64bit ? "riscv64-unknown-unknown-elf" : "riscv32-unknown-unknown-elf";
  dev->llvm_abi = is_64bit ? "lp64d" : "ilp32f";
  dev->llvm_cpu = is_64bit ? "generic-rv64" : "generic-rv32";
  dev->kernellib_name = is_64bit ? "kernel-riscv64" : "kernel-riscv32";
  dev->kernellib_fallback_name = NULL;
  dev->kernellib_subdir = "vortex";
  dev->device_aux_functions = vortex_native_device_aux_funcs;

  dev->image_support = CL_FALSE;

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

  uint64_t num_warps;
  vx_err = vx_dev_caps(vx_device, VX_CAPS_NUM_WARPS, &num_warps);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(dd);
    return CL_DEVICE_NOT_FOUND;
  }

  uint64_t num_threads;
  vx_err = vx_dev_caps(vx_device, VX_CAPS_NUM_THREADS, &num_threads);
  if (vx_err != 0) {
    vx_dev_close(vx_device);
    free(dd);
    return CL_DEVICE_NOT_FOUND;
  }

  uint64_t max_work_group_size = num_warps * num_threads;

  dev->global_mem_size = global_mem_size;
  dev->max_mem_alloc_size = global_mem_size;
  dev->local_mem_size = local_mem_size;
  dev->max_work_group_size    = max_work_group_size;
  dev->max_work_item_sizes[0] = max_work_group_size;
  dev->max_work_item_sizes[1] = max_work_group_size;
  dev->max_work_item_sizes[2] = max_work_group_size;
  dev->max_compute_units = num_cores;

  dd->vx_kernel_buffer = NULL;
  dd->vx_device = vx_device;

  dd->ctx_refcount = 0;

  dd->is_64bit = is_64bit;

  POCL_INIT_LOCK(dd->compile_lock);
  POCL_INIT_LOCK(dd->cq_lock);

  dev->data = dd;
  dev->available = &vortex_available;

  return CL_SUCCESS;
}

cl_int pocl_vortex_uninit (unsigned j, cl_device_id dev) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  if (NULL == dd)
    return CL_SUCCESS;

  if (dd->vx_kernel_buffer != NULL) {
    vx_mem_free(dd->vx_kernel_buffer);
  }
  vx_dev_close(dd->vx_device);

  POCL_DESTROY_LOCK (dd->compile_lock);
  POCL_DESTROY_LOCK (dd->cq_lock);
  POCL_MEM_FREE(dd);
  dev->data = NULL;
  return CL_SUCCESS;
}

int pocl_vortex_init_context (cl_device_id dev, cl_context context) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  if (NULL == dd)
    return CL_SUCCESS;

  dd->ctx_refcount++;

  return CL_SUCCESS;
}

int pocl_vortex_free_context (cl_device_id dev, cl_context context) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  if (NULL == dd)
    return CL_SUCCESS;

  if (--dd->ctx_refcount == 0) {
    pocl_vortex_uninit(0, dev);
  }

  return CL_SUCCESS;
}

int pocl_vortex_post_build_program (cl_program program, cl_uint device_i) {
  int result;
  cl_device_id dev = program->devices[device_i];
  vortex_device_data_t *ddata = (vortex_device_data_t *)dev->data;
  vortex_program_data_t *pdata = NULL;

  POCL_LOCK (ddata->compile_lock);

  do {
    result = pocl_llvm_run_passes_on_program (program, device_i);
    if (result != 0)
      break;

    pdata = (vortex_program_data_t *)calloc (1, sizeof (vortex_program_data_t));
    pdata->kernel_names = NULL;

    char sz_program_bc[POCL_MAX_PATHNAME_LENGTH];
    char sz_program_vxbin[POCL_MAX_PATHNAME_LENGTH];

    pocl_cache_program_bc_path(sz_program_bc, program, device_i);
    remove_extension(sz_program_bc);

    strcpy(sz_program_vxbin, sz_program_bc);
    strncat(sz_program_vxbin, ".vxbin", POCL_MAX_PATHNAME_LENGTH - 1);

    result = compile_vortex_program(&pdata->kernel_names, &pdata->num_kernels,
        sz_program_vxbin, program->llvm_irs[device_i]);
    if (result != 0)
      break;

  } while (0);

  program->data[device_i] = pdata;

  POCL_UNLOCK (ddata->compile_lock);

  return result;
}

int pocl_vortex_free_program (cl_device_id dev, cl_program program,
                              unsigned device_i) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
  vortex_program_data_t *pdata = (vortex_program_data_t *)program->data[device_i];
  if (pdata == NULL)
    return CL_SUCCESS;

  pocl_driver_free_program (dev, program, device_i);

  POCL_MEM_FREE (pdata->kernel_names);
  POCL_MEM_FREE (pdata);
  program->data[device_i] = NULL;

  return CL_SUCCESS;
}

int pocl_vortex_create_kernel (cl_device_id dev, cl_program program,
                               cl_kernel kernel, unsigned device_i) {
  int result = CL_SUCCESS;
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  vortex_kernel_data_t *kdata  = (vortex_kernel_data_t *)meta->data[device_i];
  if (kdata != NULL) {
    ++kdata->refcount;
    return CL_SUCCESS;
  }

  do {
    vortex_program_data_t *pdata = (vortex_program_data_t *)program->data[device_i];
    assert(pdata != NULL);

    const char* current = pdata->kernel_names;
    int i = 0;
    int found = 0;
    for (; i < pdata->num_kernels; ++i) {
      if (strcmp(current, kernel->name) == 0) {
        found = 1;
        break;
      }
      current += strlen(current) + 1;
    }
    assert(found);
    kdata = (void *)calloc (1, sizeof (vortex_kernel_data_t));
    kdata->kernel_id = i;
    ++kdata->refcount;

  } while (0);

  meta->data[device_i] = kdata;

  return result;
}

int pocl_vortex_free_kernel (cl_device_id dev, cl_program program,
                             cl_kernel kernel, unsigned device_i) {
  pocl_kernel_metadata_t *meta = kernel->meta;
  assert(meta->data != NULL);
  vortex_kernel_data_t *kdata = (vortex_kernel_data_t *)meta->data[device_i];
  if (kdata == NULL)
    return CL_SUCCESS;

  --kdata->refcount;
  if (kdata->refcount == 0) {
    POCL_MEM_FREE (kdata);
    meta->data[device_i] = NULL;
  }

  return CL_SUCCESS;
}

void pocl_vortex_run (void *data, _cl_command_node *cmd) {
  vortex_device_data_t *dd;
  struct pocl_argument *al;
  cl_uint device_i = cmd->program_device_i;
  cl_kernel kernel = cmd->command.run.kernel;
  cl_program program = kernel->program;
  pocl_kernel_metadata_t *meta = kernel->meta;
  vortex_program_data_t *pdata = (vortex_program_data_t *)program->data[device_i];
  vortex_kernel_data_t *kdata = (vortex_kernel_data_t *)meta->data[device_i];
  struct pocl_context *pc = &cmd->command.run.pc;
  int vx_err;

  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < pc->work_dim; ++i) {
    num_groups *= pc->num_groups[i];
    group_size *= pc->local_size[i];
  }
  if (num_groups == 0 || group_size == 0)
    return;

  assert (data != NULL);
  dd = (vortex_device_data_t *)data;

  uint32_t ptr_size = dd->is_64bit ? 8 : 4;

  uint32_t aligned_kernel_args_size = alignOffset(sizeof(kernel_args_t), ptr_size);

  // calculate kernel arguments buffer size
  uint32_t local_mem_size = 0;
  size_t abuf_size = 0;

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      local_mem_size += al->size;
      abuf_size = alignOffset(abuf_size + 4, ptr_size);
    } else
    if ((meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
     || (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
     || (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)) {
      abuf_size = alignOffset(abuf_size + ptr_size, ptr_size);
    } else {
      // scalar argument
      abuf_size = alignOffset(abuf_size + al->size, ptr_size);
    }
  }

  // local buffers
  for (int i = 0; i < meta->num_locals; ++i) {
    local_mem_size += meta->local_sizes[i];
    abuf_size = alignOffset(abuf_size + 4, ptr_size);
  }

  // add local size
  if (local_mem_size != 0) {
    abuf_size = alignOffset(abuf_size + 4, ptr_size);
  }

  // check occupancy
  if (group_size != 1) {
    int available_localmem;
    vx_err = vx_check_occupancy(dd->vx_device, group_size, &available_localmem);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }
    if (local_mem_size > available_localmem) {
      POCL_ABORT("out of local memory: needed=%d bytes, available=%d bytes\n",
        local_mem_size, available_localmem);
    }
  }

  // allocate arguments host buffer
  size_t kargs_buffer_size = aligned_kernel_args_size + abuf_size;
  uint8_t* const host_kargs_base_ptr = malloc(kargs_buffer_size);
  assert(host_kargs_base_ptr);

  // allocate kernel arguments buffer
  vx_buffer_h vx_kargs_buffer;
  vx_err = vx_mem_alloc(dd->vx_device, kargs_buffer_size, VX_MEM_READ, &vx_kargs_buffer);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  uint64_t dev_kargs_base_addr;
  vx_err = vx_mem_address(vx_kargs_buffer, &dev_kargs_base_addr);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // write context data
  {
    kernel_args_t* const kargs = (kernel_args_t*)host_kargs_base_ptr;
    kargs->work_dim = pc->work_dim;
    for (int i = 0; i < 3; ++i) {
      kargs->num_groups[i] = pc->num_groups[i];
      kargs->local_size[i] = pc->local_size[i];
      kargs->global_offset[i] = pc->global_offset[i];
    }
    kargs->kernel_id = kdata->kernel_id;
  }

  // write arguments

  uint8_t* const host_args_ptr = host_kargs_base_ptr + aligned_kernel_args_size;
  uint32_t host_args_offset = 0;
  uint32_t local_mem_offset = 0;

  for (int i = 0; i < meta->num_args; ++i) {
    struct pocl_argument* al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      if (local_mem_offset == 0) {
        memcpy(host_args_ptr + host_args_offset, &local_mem_size, 4); // local_size
        host_args_offset = alignOffset(host_args_offset + 4, ptr_size);
      }
      memcpy(host_args_ptr + host_args_offset, &local_mem_offset, 4); // arg offset
      host_args_offset = alignOffset(host_args_offset + 4, ptr_size);
      local_mem_offset += al->size;
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      if (al->value == NULL) {
        memset(host_args_ptr + host_args_offset, 0, ptr_size); // NULL pointer value
        host_args_offset = alignOffset(host_args_offset + ptr_size, ptr_size);
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        vortex_buffer_data_t* buf_data = (vortex_buffer_data_t *) m->device_ptrs[cmd->device->global_mem_id].mem_ptr;
        uint64_t dev_mem_addr = buf_data->buf_address + al->offset;
        memcpy(host_args_ptr + host_args_offset, &buf_data->buf_address, ptr_size); // pointer value
        host_args_offset = alignOffset(host_args_offset + ptr_size, ptr_size);
      }
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
        POCL_ABORT("POCL_VORTEX_RUN\n");
    } else
    if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
        POCL_ABORT("POCL_VORTEX_RUN\n");
    } else {
      // scalar argument
      memcpy(host_args_ptr + host_args_offset, al->value, al->size); // scalar value
      host_args_offset = alignOffset(host_args_offset + al->size, ptr_size);
    }
  }

  // write local arguments
  for (int i = 0; i < meta->num_locals; ++i) {
    if (local_mem_offset == 0) {
      memcpy(host_args_ptr + host_args_offset, &local_mem_size, 4); // local_size
      host_args_offset = alignOffset(host_args_offset + 4, ptr_size);
    }
    memcpy(host_args_ptr + host_args_offset, &local_mem_offset, 4); // arg offset
    host_args_offset = alignOffset(host_args_offset + 4, ptr_size);
    local_mem_offset += meta->local_sizes[i];
  }

  // upload kernel arguments buffer
  vx_err = vx_copy_to_dev(vx_kargs_buffer, host_kargs_base_ptr, 0, kargs_buffer_size);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // release argument host buffer
  free(host_kargs_base_ptr);

  // upload kernel to device
  if (NULL == dd->vx_kernel_buffer) {
    char sz_program_bc[POCL_MAX_PATHNAME_LENGTH];
    char sz_program_vxbin[POCL_MAX_PATHNAME_LENGTH];

    pocl_cache_program_bc_path(sz_program_bc, program, device_i);
    remove_extension(sz_program_bc);

    strcpy(sz_program_vxbin, sz_program_bc);
    strncat(sz_program_vxbin, ".vxbin", POCL_MAX_PATHNAME_LENGTH - 1);

    vx_err = vx_upload_kernel_file(dd->vx_device, sz_program_vxbin, &dd->vx_kernel_buffer);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
    }
  }

  // launch kernel execution
  vx_err = vx_start(dd->vx_device, dd->vx_kernel_buffer, vx_kargs_buffer);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // wait for the execution to complete
  vx_err = vx_ready_wait(dd->vx_device, -1);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_RUN\n");
  }

  // release arguments device buffer
  vx_mem_free(vx_kargs_buffer);
}

cl_int pocl_vortex_alloc_mem_obj(cl_device_id dev, cl_mem mem_obj, void *host_ptr) {
  int vx_err;
  pocl_mem_identifier *p = &mem_obj->device_ptrs[dev->global_mem_id];

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

    vortex_device_data_t* dd = (vortex_device_data_t *)dev->data;

    vx_buffer_h vx_buffer;
    vx_err = vx_mem_alloc(dd->vx_device, mem_obj->size, vx_flags, &vx_buffer);
    if (vx_err != 0) {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }

    uint64_t buf_address;
    vx_err = vx_mem_address(vx_buffer, &buf_address);
    if (vx_err != 0) {
      POCL_ABORT("POCL_VORTEX_RUN\n");
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

    vortex_buffer_data_t* buf_data = (vortex_buffer_data_t *)malloc(sizeof(vortex_buffer_data_t));
    buf_data->vx_device = dd->vx_device;
    buf_data->vx_buffer = vx_buffer;
    buf_data->buf_address = buf_address;

    p->mem_ptr = buf_data;
  }

  return CL_SUCCESS;
}

void pocl_vortex_free(cl_device_id dev, cl_mem mem_obj) {
  pocl_mem_identifier *p = &mem_obj->device_ptrs[dev->global_mem_id];
  cl_mem_flags flags = mem_obj->flags;
  vortex_buffer_data_t* buf_data = (vortex_buffer_data_t*)p->mem_ptr;

  if (flags & CL_MEM_USE_HOST_PTR) {
    POCL_ABORT("POCL_VORTEX_FREE\n");
  } else {
    if (flags & CL_MEM_ALLOC_HOST_PTR) {
      pocl_release_mem_host_ptr(mem_obj);
    }
    if (buf_data->vx_buffer) {
      vx_mem_free(buf_data->vx_buffer);
    }
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
  vortex_buffer_data_t *buf_data = (vortex_buffer_data_t *)dst_mem_id->mem_ptr;
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
  vortex_buffer_data_t* buf_data = (vortex_buffer_data_t*)src_mem_id->mem_ptr;
  vx_err = vx_copy_from_dev(host_ptr, buf_data->vx_buffer, offset, size);
  if (vx_err != 0) {
    POCL_ABORT("POCL_VORTEX_READ\n");
  }
}

static void vortex_command_scheduler (vortex_device_data_t *dd) {
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

void pocl_vortex_submit (_cl_command_node *node, cl_command_queue cq) {
  vortex_device_data_t *dd = (vortex_device_data_t *)node->device->data;

  node->ready = 1;
  POCL_LOCK (dd->cq_lock);
  pocl_command_push(node, &dd->ready_list, &dd->command_list);

  POCL_UNLOCK_OBJ (node->sync.event.event);
  vortex_command_scheduler (dd);
  POCL_UNLOCK (dd->cq_lock);

  return;
}

void pocl_vortex_flush (cl_device_id dev, cl_command_queue cq) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;

  POCL_LOCK (dd->cq_lock);
  vortex_command_scheduler (dd);
  POCL_UNLOCK (dd->cq_lock);
}

void pocl_vortex_join (cl_device_id dev, cl_command_queue cq) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;

  POCL_LOCK (dd->cq_lock);
  vortex_command_scheduler (dd);
  POCL_UNLOCK (dd->cq_lock);

  return;
}

void pocl_vortex_notify (cl_device_id dev, cl_event event, cl_event finished) {
  vortex_device_data_t *dd = (vortex_device_data_t *)dev->data;
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