#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdarg>
#include <vector>
#include <cstdio>
#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

#include "devices/vortex/pocl-vortex-config.h"
#include "pocl.h"
#include "pocl_llvm_api.h"
#include "pocl_runtime_config.h"
#include "pocl_file_util.h"
#include "pocl_cache.h"
#include "LLVMUtils.h"
#include "pocl_util.h"


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
      //throw std::runtime_error("popen() failed!");
      return -1;
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
                                   const char *kernel_out,
                                   int specialize) {
  cl_program program = kernel->program;

  std::string kernel_bc_s(kernel_bc);
  std::string kernel_obj_s(kernel_obj);
  std::string kerenl_out_s(kernel_out);
  std::string kernel_elf_s, wrapper_cc_s;

  std::string build_cflags = pocl_get_string_option ("POCL_VORTEX_CFLAGS", "");
  if(build_cflags == ""){
    POCL_MSG_ERR("LLVM_PREFIX : 'POCL_VORTEX_CFLAGS' need to be set\n");
    return -1;
  }

  std::string build_ldflags = pocl_get_string_option ("POCL_VORTEX_LDFLAGS", "");
  if(build_ldflags == ""){
    POCL_MSG_ERR("LLVM_PREFIX : 'POCL_VORTEX_LDFLAGS' need to be set\n");
    return -1;
  }

  std::string build_schedule = pocl_get_string_option ("VORTEX_SCHEDULE_FLAG", "0");
  int schedule_flag = std::stoi(build_schedule);

  const char* llvm_install_path = getenv("LLVM_PREFIX");
  if (llvm_install_path) {
    if (!pocl_exists(llvm_install_path)) {
      POCL_MSG_ERR("$LLVM_PREFIX: '%s' doesn't exist\n", llvm_install_path);
      return -1;
    }
    POCL_MSG_PRINT_INFO("using $LLVM_PREFIX=%s!\n", llvm_install_path);
  }

  std::string llvm_prefix(LLVM_PREFIX);
  if (llvm_install_path) {
    llvm_prefix = llvm_install_path;
  }

  {
    std::stringstream ss_cmd, ss_out;
    ss_cmd << llvm_prefix << "/bin/llvm-dis " << kernel_bc << " -o " << kernel->name << ".ll";
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  std::string clang_path(CLANG);
  if (llvm_install_path) {
    clang_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path);
  }

  {
    std::stringstream ss_cmd, ss_out;
    ss_cmd << clang_path.c_str() << " " << build_cflags << " " << kernel_bc << " -c -o " << kernel_obj;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  #ifndef NDEBUG
    POCL_MSG_PRINT_LLVM("%s\n", ss_out.str().c_str());
  #endif
  }

  {
   /*    SW warp scheduling, Mapping one SW warp to HW.
    TM (0):   mapping one SW warp to one HW thread, vx_spawn_kernel
    CM (1):   mapping one SW warp to one HW core, vx_spawn_kernel_cm
    GM (2):   mapping one SW warp to N number of HW cores, vx_spawn_kernel_gm
    GDM (3):  mapping one SW warp to N number of HW cores with M group size of SW thread for the distribution, vx_spwan_kernel_gdm
    PM (4):   mapping one SW warp to N number of HW cores with N pipelining method, vx_spwan_kernel_pm
    */

    char pfn_workgroup_string[WORKGROUP_STRING_LENGTH];
    std::stringstream ss;

    snprintf (pfn_workgroup_string, WORKGROUP_STRING_LENGTH, "_pocl_kernel_%s_workgroup", kernel->name);

    ss << "#include <vx_intrinsics.h>\n"
          "#include <vx_spawn.h>\n"
          "\n"
          "typedef struct {\n"
          "  uint32_t num_groups[3];\n"
          "  uint32_t global_offset[3];\n"
          "  uint32_t local_size[3];\n"
          "  uint32_t printf_buffer;\n"
          "  uint32_t printf_buffer_position;\n"
          "  uint32_t printf_buffer_capacity;\n"
          "  uint32_t work_dim;\n"
          "} pocl_kernel_context_t;\n"
          "\n"
          "typedef void (*pocl_kernel_cb) (\n"
          "  const void * arg,\n"
          "  const pocl_kernel_context_t * context,\n"
          "  uint32_t group_x,\n"
          "  uint32_t group_y,\n"
          "  uint32_t group_z,\n"
          "  uint32_t local_offset\n"
          ");\n"
          "typedef struct {\n"
          "  pocl_kernel_context_t * ctx;\n"
          "  pocl_kernel_cb callback;\n"
          "  void* arg;\n"
          "  int  local_size;\n"
          "  int  isXYpow2;\n"
          "  int  lgXY;\n"
          "  int  lgX;\n"
          "} wspawn_pocl_kernel_args_t;\n"
          "\n"
          "inline char is_log2(int x) {\n"
          "  return ((x & (x-1)) == 0);\n"
          "}\n"
          "\n"
          "inline int log2_fast(int x) {\n"
          "  return 31 - __builtin_clz (x);\n"
          "}\n"
          "\n"
          "static void __attribute__ ((noinline)) spawn_pocl_kernel_cb(int task_id, void *arg) {\n"
          "  wspawn_pocl_kernel_args_t* p_wspawn_args = (wspawn_pocl_kernel_args_t*)arg;\n"
          "  pocl_kernel_context_t* pocl_ctx = p_wspawn_args->ctx;\n"
          "  void* pocl_arg = p_wspawn_args->arg;\n"
          "  if (p_wspawn_args->isXYpow2) {\n"
          "    int k = task_id >> p_wspawn_args->lgXY;\n"
          "    int wg_2d = task_id - (k << p_wspawn_args->lgXY);\n"
          "    int j = wg_2d >> p_wspawn_args->lgX;\n"
          "    int i = wg_2d - (j << p_wspawn_args->lgX);\n"
          "    int local_offset = task_id * p_wspawn_args->local_size;\n"
          "    (p_wspawn_args->callback)(pocl_arg, pocl_ctx, i, j, k, local_offset);\n"
          "  } else {\n"
          "    int X = pocl_ctx->num_groups[0];\n"
          "    int Y = pocl_ctx->num_groups[1];\n"
          "    int XY = X * Y;\n"
          "    int k = task_id / XY;\n"
          "    int wg_2d = task_id - k * XY;\n"
          "    int j = wg_2d / X;\n"
          "    int i = wg_2d - j * X;\n"
          "    int local_offset = task_id * p_wspawn_args->local_size;\n"
          "    (p_wspawn_args->callback)(pocl_arg, pocl_ctx, i, j, k, local_offset);\n"
          "  }\n"
          "}\n"
          "void " << pfn_workgroup_string << "(uint8_t* args, uint8_t* ctx, uint32_t group_x, uint32_t group_y, uint32_t group_z, uint32_t local_offset);\n"
          "\n"
          "int main() {\n"
          "  pocl_kernel_context_t* ctx = (pocl_kernel_context_t*)csr_read(VX_CSR_MSCRATCH);\n"
          "  void* args = (void*)((uint8_t*)ctx + " << ALIGNED_CTX_SIZE << ");\n";

    if (schedule_flag == 1){
    ss << "  vx_spawn_kernel_cm(ctx, (void*)" << pfn_workgroup_string << ", args);\n";
    } else {
    ss << "  int X = ctx->num_groups[0];\n"
          "  int Y = ctx->num_groups[1];\n"
          "  int Z = ctx->num_groups[2];\n"
          "  int XY = X * Y;\n"
          "  int num_tasks = XY * Z;\n"
          "\n"
          "  char isXYpow2 = is_log2(XY);\n"
          "  char lgXY = log2_fast(XY);\n"
          "  char lgX = log2_fast(X);\n"
          "\n"
          "  int local_size = ctx->local_size[0] * ctx->local_size[1] * ctx->local_size[2];\n"
          "\n"
          "  wspawn_pocl_kernel_args_t kernel_args = {ctx, (void*)" << pfn_workgroup_string << ", args, local_size, isXYpow2, lgXY, lgX};\n"
          "  vx_spawn_tasks(num_tasks, spawn_pocl_kernel_cb, &kernel_args);\n";
    }
    ss << "  return 0;\n"
          "}";

    {
      char wrapper_cc[POCL_MAX_PATHNAME_LENGTH + 1];
      auto content = ss.str();
      int err = pocl_write_tempfile(wrapper_cc, "/tmp/pocl_vortex_kernel", ".c",
                              content.c_str(), content.size(), nullptr);
      if (err != 0)
        return err;

      wrapper_cc_s = std::string(wrapper_cc);
    }
    {
      char kernel_elf[POCL_MAX_PATHNAME_LENGTH + 1];
      int err = pocl_mk_tempname(kernel_elf, "/tmp/pocl_vortex_kernel", ".elf", nullptr);
      if (err != 0)
        return err;
      kernel_elf_s = std::string(kernel_elf);
    }
    StaticStrFormat ssfmt(9);

    {
      std::stringstream ss_cmd, ss_out;
      ss_cmd << clang_path.c_str() << " " << build_cflags << " -I" POCL_INSTALL_PRIVATE_HEADER_DIR << " "<< wrapper_cc_s << " " << kernel_obj_s << " " << build_ldflags << " -o " << kernel_elf_s;
      POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
      int err = exec(ss_cmd.str().c_str(), ss_out);
      if (err != 0) {
        POCL_MSG_ERR("%s\n", ss_out.str().c_str());
        return err;
      }
    }
  }

  {
    std::string vxbin_path = pocl_get_string_option ("POCL_VORTEX_BINTOOL", "");
    if (vxbin_path == ""){
      POCL_MSG_ERR("LLVM_PREFIX : 'POCL_VORTEX_BINTOOL' need to be set\n");
      return -1;
    }
    std::stringstream ss_cmd, ss_out;
    ss_cmd << vxbin_path << " " << kernel_elf_s << " " << kernel_out;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
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
    ss_cmd << objdump_path.c_str() << " -D " << kernel_elf_s << " > " << kernel->name;
    if (specialize != 0) {
      ss_cmd << "_op" << specialize;
    }
    ss_cmd << ".dump";

    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  return 0;
}
