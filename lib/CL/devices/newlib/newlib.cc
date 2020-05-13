/* newlib.c - a minimalistic pocl device driver layer implementation

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

#include "newlib.h"
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

#include "pocl_cache.h"
#include "pocl_file_util.h"
#include "pocl_timing.h"
#include "pocl_workgroup_func.h"

#ifdef OCS_AVAILABLE
#include "pocl_llvm.h"
#endif

#include "prototypes.inc"
GEN_PROTOTYPES (basic)

#ifdef __cplusplus
}
#endif

#include <sstream>

/* Maximum kernels occupancy */
#define MAX_KERNELS 16

/* default WG size in each dimension & total WG size.
 * this should be reasonable for CPU */
#define DEFAULT_WG_SIZE 4096

struct data {
  /* Currently loaded kernel. */
  cl_kernel current_kernel;

  /* List of commands ready to be executed */
  _cl_command_node *volatile ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node *volatile command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;
  /* printf buffer */
  void *printf_buffer;
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

#if !defined(OCS_AVAILABLE)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  const char* name;
  const void* pfn;
  uint32_t num_args;
  uint32_t num_locals;
  const uint8_t* arg_types;
  const uint32_t* local_sizes;
} kernel_info_t;

static int g_num_kernels = 0;
static kernel_info_t g_kernels [MAX_KERNELS];

int _pocl_register_kernel(const char* name, const void* pfn, uint32_t num_args, uint32_t num_locals, const uint8_t* arg_types, const uint32_t* local_sizes) {
  //printf("******** _pocl_register_kernel\n");
  //printf("Name to register: %s\n", name);
  //printf("PTR of name: %x\n", name);
  if (g_num_kernels == MAX_KERNELS)
  {
    //printf("ERROR: REACHED MAX KERNELS\n");
    return -1;  
  }

  //printf("Going to register at index: %d\n", g_num_kernels);

  kernel_info_t* kernel = g_kernels + g_num_kernels++;
  kernel->name = name;
  kernel->pfn = pfn;
  kernel->num_args = num_args;
  kernel->num_locals = num_locals;
  kernel->arg_types = arg_types;
  kernel->local_sizes = local_sizes;
  //printf("New kernel name: %s\n", kernel->name);
  return 0;
}

int _pocl_query_kernel(const char* name, const void** p_pfn, uint32_t* p_num_args, uint32_t* p_num_locals, const uint8_t** p_arg_types, const uint32_t** p_local_sizes) {
  //printf("********* Inside _pocl_query_kernel\n");
  //printf("name: %s\n", name);
  //printf("g_num_kernels: %d\n", g_num_kernels);
  for (int i = 0; i < g_num_kernels; ++i) {
    //printf("Currently quering index %d\n", i);
    kernel_info_t* kernel = g_kernels + i;
    if (strcmp(kernel->name, name) != 0)
    {
      //printf("STR CMP failed! kernel->name = %s \t name: %s\n", kernel->name, name);
      continue;
    }
    //printf("!!!!!!!!!STR CMP PASSED\n");
    if (p_pfn) *p_pfn = kernel->pfn;
    if (p_num_args) *p_num_args = kernel->num_args;
    if (p_num_locals) *p_num_locals = kernel->num_locals;
    if (p_arg_types) *p_arg_types = kernel->arg_types;
    if (p_local_sizes) *p_local_sizes = kernel->local_sizes;
    return 0;
  }
  return -1;
}

cl_int pocl_newlib_supports_builtin_kernel(void *data,
                                                  const char *kernel_name) {
  int err = _pocl_query_kernel(kernel_name, NULL, NULL, NULL, NULL, NULL);
  if (0 == err)
    return 1;
  return 0;
}

cl_int pocl_newlib_get_builtin_kernel_metadata(void *data, const char *kernel_name,
                                        pocl_kernel_metadata_t *target) {
  const void *pfn;
  uint32_t num_args, num_locals;
  const uint8_t *arg_types;
  const uint32_t *local_sizes;

  int err = _pocl_query_kernel(kernel_name, &pfn, &num_args, &num_locals,
                               &arg_types, &local_sizes);
  if (err)
    err;

  target->name = strdup(kernel_name);
  target->data = (void **)calloc(1, sizeof(void *));
  target->data[0] = (void*)pfn;
  target->num_args = num_args;
  target->num_locals = num_locals;

  if (num_args) {
    target->arg_info = (struct pocl_argument_info *)calloc(
        num_args, sizeof(struct pocl_argument_info));
    for (uint32_t i = 0; i < num_args; ++i) {      
      if (arg_types[i] == 4) {
        target->arg_info[i].address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
        target->arg_info[i].type = POCL_ARG_TYPE_NONE;
      } else {
        target->arg_info[i].type = (pocl_argument_type)arg_types[i];
      }
    }
  }

  if (num_locals) {
    target->local_sizes = (size_t *)calloc(num_locals, sizeof(size_t));
    for (uint32_t i = 0; i < num_locals; ++i) {
      target->local_sizes[i] = local_sizes[i];
    }
  }

  return 0;
}

#ifdef __cplusplus
}
#endif

#endif

void pocl_newlib_init_device_ops(struct pocl_device_ops *ops) {
  pocl_basic_init_device_ops(ops);

  ops->device_name = "newlib";
  ops->probe = pocl_newlib_probe;
  ops->uninit = pocl_newlib_uninit;
  ops->reinit = pocl_newlib_reinit;
  ops->init = pocl_newlib_init;
  ops->build_hash = pocl_newlib_build_hash;

#if !defined(OCS_AVAILABLE)
  ops->supports_builtin_kernel = pocl_newlib_supports_builtin_kernel;
  ops->get_builtin_kernel_metadata = pocl_newlib_get_builtin_kernel_metadata;
#endif

  ops->compile_kernel = pocl_newlib_compile_kernel;

  ops->run = pocl_newlib_run;
  ops->run_native = pocl_newlib_run_native;

  ops->svm_free = pocl_newlib_svm_free;
  ops->svm_alloc = pocl_newlib_svm_alloc;
  /* no need to implement these two as they're noop
   * and pocl_exec_command takes care of it */
  ops->svm_map = NULL;
  ops->svm_unmap = NULL;
  ops->svm_copy = pocl_newlib_svm_copy;
  ops->svm_fill = pocl_newlib_svm_fill;
}

char *pocl_newlib_build_hash(cl_device_id device) {
  char *res = (char*)calloc(1000, sizeof(char));
#ifdef KERNELLIB_HOST_DISTRO_VARIANTS
  char *name = get_llvm_cpu_name();
  snprintf(res, 1000, "newlib-%s-%s", HOST_DEVICE_BUILD_HASH, name);
  POCL_MEM_FREE(name);
#else
  snprintf(res, 1000, "newlib-%s", HOST_DEVICE_BUILD_HASH);
#endif
  return res;
}

static cl_device_partition_property newlib_partition_properties[1] = {0};

static const char *final_ld_flags[] = {"-nostartfiles", HOST_LD_FLAGS_ARRAY, NULL};

unsigned int pocl_newlib_probe(struct pocl_device_ops *ops) {
  if (0 == strcmp(ops->device_name, "newlib"))
    return 1;
  return 0;
}

cl_int pocl_newlib_init(unsigned j, cl_device_id device,
                        const char *parameters) {
  struct data *d;
  cl_int ret = CL_SUCCESS;
  int err;
  static int first_newlib_init = 1;

  if (first_newlib_init) {
    POCL_MSG_WARN("INIT dlcache DOTO delete\n");
    pocl_init_dlhandle_cache();
    first_newlib_init = 0;
  }
  device->global_mem_id = 0;

  d = (struct data *)calloc(1, sizeof(struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;
  device->data = d;

  pocl_init_cpu_device_infos(device);

  device->llvm_cpu = "";

#if defined(OCS_AVAILABLE)
  /* hwloc probes OpenCL device info at its initialization in case
     the OpenCL extension is enabled. This causes to printout
     an unimplemented property error because hwloc is used to
     initialize global_mem_size which it is not yet. Just put
     a nonzero there for now. */
  device->global_mem_size = 1;
  err = pocl_topology_detect_device_info(device);
  if (err)
    ret = CL_INVALID_DEVICE;
#else
  device->global_mem_size = 2 * MIN_MAX_MEM_ALLOC_SIZE;
  device->global_mem_cache_size = 1;
  device->max_compute_units = 1;
#endif

  POCL_INIT_LOCK(d->cq_lock);

  assert(device->printf_buffer_size > 0);
  d->printf_buffer =
      pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, device->printf_buffer_size);
  assert(d->printf_buffer != NULL);

  pocl_cpuinfo_detect_device_info(device);
  pocl_set_buffer_image_limits(device);

  /* in case hwloc doesn't provide a PCI ID, let's generate
     a vendor id that hopefully is unique across vendors. */
  const char *magic = "pocl";
  if (device->vendor_id == 0)
    device->vendor_id =
        magic[0] | magic[1] << 8 | magic[2] << 16 | magic[3] << 24;

  device->vendor_id += j;

  /* The newlib driver represents only one "compute unit" as
     it doesn't exploit multiple hardware threads. Multiple
     newlib devices can be still used for task level parallelism
     using multiple OpenCL devices. */
  device->max_compute_units = 1;

  return ret;
}

void pocl_newlib_run(void *data, _cl_command_node *cmd) {
  struct data *d;
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;

  assert(data != NULL);
  d = (struct data *)data;

  d->current_kernel = kernel;

  void **arguments =
      (void **)malloc(sizeof(void *) * (meta->num_args + meta->num_locals));

  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
  for (i = 0; i < meta->num_args; ++i) {
    al = &(cmd->command.run.arguments[i]);
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      if (cmd->device->device_alloca_locals) {
        /* Local buffers are allocated in the device side work-group
           launcher. Let's pass only the sizes of the local args in
           the arg buffer. */
        assert(sizeof(size_t) == sizeof(void *));
        arguments[i] = (void *)(uintptr_t)al->size;
      } else {
        arguments[i] = malloc(sizeof(void *));
        *(void **)(arguments[i]) =
            pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, al->size);
      }
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      /* It's legal to pass a NULL pointer to clSetKernelArguments. In
         that case we must pass the same NULL forward to the kernel.
         Otherwise, the user must have created a buffer with per device
         pointers stored in the cl_mem. */
      arguments[i] = malloc(sizeof(void *));
      if (al->value == NULL) {
        *(void **)arguments[i] = NULL;
      } else {
        cl_mem m = (*(cl_mem *)(al->value));
        void *ptr = m->device_ptrs[cmd->device->dev_id].mem_ptr;
        *(void **)arguments[i] = (char *)ptr + al->offset;
      }
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE) {
      dev_image_t di;
      fill_dev_image_t(&di, al, cmd->device);

      void *devptr =
          pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, sizeof(dev_image_t));
      arguments[i] = malloc(sizeof(void *));
      *(void **)(arguments[i]) = devptr;
      memcpy(devptr, &di, sizeof(dev_image_t));
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      dev_sampler_t ds;
      fill_dev_sampler_t(&ds, al);
      arguments[i] = malloc(sizeof(void *));
      *(void **)(arguments[i]) = (void *)ds;
    } else {
      arguments[i] = al->value;
    }
  }

  if (!cmd->device->device_alloca_locals)
    for (i = 0; i < meta->num_locals; ++i) {
      size_t s = meta->local_sizes[i];
      size_t j = meta->num_args + i;
      arguments[j] = malloc(sizeof(void *));
      void *pp = pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, s);
      *(void **)(arguments[j]) = pp;
    }

  pc->printf_buffer = (uchar*)d->printf_buffer;
  assert(pc->printf_buffer != NULL);
  pc->printf_buffer_capacity = cmd->device->printf_buffer_size;
  assert(pc->printf_buffer_capacity > 0);
  uint position = 0;
  pc->printf_buffer_position = &position;

  unsigned rm = pocl_save_rm();
  pocl_set_default_rm();
  unsigned ftz = pocl_save_ftz();
  pocl_set_ftz(kernel->program->flush_denorms);  
  
#if (HOST_DEVICE_ADDRESS_BITS==32)
  auto pfn = (pocl_workgroup_func32)cmd->command.run.wg;
 #else
  auto pfn = (pocl_workgroup_func)cmd->command.run.wg;
#endif

  for (z = 0; z < pc->num_groups[2]; ++z) {
    for (y = 0; y < pc->num_groups[1]; ++y) {
      for (x = 0; x < pc->num_groups[0]; ++x) {
        (pfn)((uchar*)arguments, (uint8_t *)pc, x, y, z);
      }
    }
  }

  pocl_restore_rm(rm);
  pocl_restore_ftz(ftz);

  if (position > 0) {
    write(STDOUT_FILENO, pc->printf_buffer, position);
    position = 0;
  }

  for (i = 0; i < meta->num_args; ++i) {
    if (ARG_IS_LOCAL(meta->arg_info[i])) {
      if (!cmd->device->device_alloca_locals) {
        POCL_MEM_FREE(*(void **)(arguments[i]));
        POCL_MEM_FREE(arguments[i]);
      } else {
        /* Device side local space allocation has deallocation via stack
           unwind. */
      }
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE ||
               meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER) {
      if (meta->arg_info[i].type != POCL_ARG_TYPE_SAMPLER)
        POCL_MEM_FREE(*(void **)(arguments[i]));
      POCL_MEM_FREE(arguments[i]);
    } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
      POCL_MEM_FREE(arguments[i]);
    }
  }

  if (!cmd->device->device_alloca_locals)
    for (i = 0; i < meta->num_locals; ++i) {
      POCL_MEM_FREE(*(void **)(arguments[meta->num_args + i]));
      POCL_MEM_FREE(arguments[meta->num_args + i]);
    }
  free(arguments);

  pocl_release_dlhandle_cache(cmd);
}

void pocl_newlib_run_native(void *data, _cl_command_node *cmd) {
  cmd->command.native.user_func(cmd->command.native.args);
}

cl_int pocl_newlib_uninit(unsigned j, cl_device_id device) {
  struct data *d = (struct data *)device->data;
  POCL_DESTROY_LOCK(d->cq_lock);
  pocl_aligned_free(d->printf_buffer);
  POCL_MEM_FREE(d);
  device->data = NULL;
  return CL_SUCCESS;
}

cl_int pocl_newlib_reinit(unsigned j, cl_device_id device) {
  struct data *d = (struct data *)calloc(1, sizeof(struct data));
  if (d == NULL)
    return CL_OUT_OF_HOST_MEMORY;

  d->current_kernel = NULL;

  assert(device->printf_buffer_size > 0);
  d->printf_buffer =
      pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, device->printf_buffer_size);
  assert(d->printf_buffer != NULL);

  POCL_INIT_LOCK(d->cq_lock);
  device->data = d;
  return CL_SUCCESS;
}

void pocl_newlib_compile_kernel(_cl_command_node *cmd, cl_kernel kernel,
                                cl_device_id device, int specialize) {
  if (cmd != NULL && cmd->type == CL_COMMAND_NDRANGE_KERNEL)
    pocl_check_kernel_dlhandle_cache(cmd, 0, specialize);
}

void pocl_newlib_svm_free(cl_device_id dev, void *svm_ptr) {
  /* TODO we should somehow figure out the size argument
   * and call pocl_free_global_mem */
  pocl_aligned_free(svm_ptr);
}

void *pocl_newlib_svm_alloc(cl_device_id dev, cl_svm_mem_flags flags,
                            size_t size) {
  return pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, size);
}

void pocl_newlib_svm_copy(cl_device_id dev, void *__restrict__ dst,
                          const void *__restrict__ src, size_t size) {
  memcpy(dst, src, size);
}

void pocl_newlib_svm_fill(cl_device_id dev, void *__restrict__ svm_ptr,
                          size_t size, void *__restrict__ pattern,
                          size_t pattern_size) {
  pocl_mem_identifier temp;
  temp.mem_ptr = svm_ptr;
  pocl_basic_memfill(dev->data, &temp, NULL, size, 0, pattern, pattern_size);
}

#if defined(OCS_AVAILABLE)

int pocl_llvm_build_newlib_program(cl_kernel kernel, 
                                   unsigned device_i, 
                                   cl_device_id device,
                                   const char *kernel_obj,
                                   char *kernel_out) {
  char wrapper_cc[POCL_FILENAME_LENGTH];
  char wrapper_obj[POCL_FILENAME_LENGTH];
  int err;
  cl_program program = kernel->program;

  {  
    char pfn_workgroup_string[WORKGROUP_STRING_LENGTH];
    std::stringstream ss;

    snprintf (pfn_workgroup_string, WORKGROUP_STRING_LENGTH,
              "_pocl_kernel_%s_workgroup", kernel->name);
    
    ss << "#include <cstdint>\n"   
          "#ifdef __cplusplus\n"
          "extern \"C\" {\n"
          "#endif\n"
          "int _pocl_register_kernel(const char* name, const void* pfn, uint32_t num_args, uint32_t num_locals, const uint8_t* arg_types, const uint32_t* local_sizes);\n"     
          "void " << pfn_workgroup_string << "(uint8_t* args, uint8_t*, uint32_t, uint32_t, uint32_t);\n"  
          "#ifdef __cplusplus\n"
          "}\n"
          "#endif\n"
          "namespace {\n"
          "class auto_register_kernel_t {\n"
          "public:\n" 
          "  auto_register_kernel_t() {\n"
          "    static uint8_t arg_types[] = {";  

    for (cl_uint i = 0; i < kernel->meta->num_args; ++i) {
      if (i) ss << ", ";
      uint32_t value = POCL_ARG_TYPE_NONE;
      if (ARG_IS_LOCAL(kernel->meta->arg_info[i]))
        value = 4;
      else
        value = kernel->meta->arg_info[i].type;
      ss << value;
    }
    ss << "};\n"; 

    ss << "    static uint32_t local_sizes[] = {";
      for (cl_uint i = 0; i < kernel->meta->num_locals; ++i) {
        if (i) ss << ", ";
        ss << kernel->meta->local_sizes[i];
      }
    ss << "};\n"; 

    ss << "    _pocl_register_kernel(\"";
    ss << kernel->name << "\", " 
      << "(void*)" << pfn_workgroup_string << ", " 
      << kernel->meta->num_args << ", " 
      << kernel->meta->num_locals << ", " 
      << "arg_types, local_sizes);\n"
          "  }\n"
          "};\n"
          "static auto_register_kernel_t __x__;\n"
          "}";

    auto content = ss.str();
    err = pocl_write_tempfile(wrapper_cc, "/tmp/pocl_register_kernel", ".cc",
                              content.c_str(), content.size(), nullptr);
    if (err)
      return err;
  } 

  {
    err = pocl_mk_tempname(wrapper_obj, "/tmp/pocl_register_kernel", ".o", nullptr);
    if (err)
      return err;

    std::stringstream march_ss;
    march_ss << "-march=" << OCL_KERNEL_TARGET_CPU;

    const char *cmd_args[] = {CLANG, march_ss.str().c_str(), "-fno-rtti" ,"-fno-exceptions", wrapper_cc, "-c", "-o", wrapper_obj, nullptr};
    err = pocl_invoke_clang(device, cmd_args);
    if (err)
      return err;
  }

  if (program->pocl_binaries[device_i]) {
    err = pocl_write_file(kernel_out, (char*)program->pocl_binaries[device_i], 
                          program->pocl_binary_sizes[device_i], 0, 0);
    if (err) {
      POCL_MSG_PRINT_LLVM ("dumping previous library to file failed\n");
      return err;
    }
    free(program->pocl_binaries[device_i]);
    program->pocl_binary_sizes[device_i] = 0;
  }

  {
    std::stringstream ss;
    ss << LLVM_AR << " rc " << kernel_out << " " << kernel_obj << " " << wrapper_obj;
    std::string s = ss.str();
    err = system(s.c_str());
    if (err)
      return err;
  }
  
  return 0;
}

#endif