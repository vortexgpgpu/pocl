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
                                   char *kernel_out) {
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

  const char* llvm_install_path = getenv("LLVM_PREFIX");
  if (llvm_install_path) {
    if (!pocl_exists(llvm_install_path)) {
      POCL_MSG_ERR("$LLVM_PREFIX: '%s' doesn't exist\n", llvm_install_path);
      return -1;    
    }
    POCL_MSG_PRINT_INFO("using $LLVM_PREFIX=%s!\n", llvm_install_path);
  }  

  std::string clang_path(CLANG);

  //if (llvm_install_path) {
  //  clang_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
  //}

  {
    std::stringstream ss_cmd, ss_out;
    ss_cmd << clang_path.c_str() << " " << build_cflags << " " << kernel_bc << " -c -o " << kernel_obj;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }
  
  {  
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
    std::string objcopy_path(LLVM_OBJCOPY);
    //if (llvm_install_path) {
    //  objcopy_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
    //}

    std::stringstream ss_cmd, ss_out;
    ss_cmd << objcopy_path.c_str() << " -O binary " << kernel_elf_s << " " << kernel_out;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  {
    std::string objdump_path(LLVM_OBJDUMP);
    //if (llvm_install_path) {
    //  objdump_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path); 
    //}

    std::stringstream ss_cmd, ss_out;
    ss_cmd << objdump_path.c_str() << " -D " << kernel_elf_s << " > " << kernel->name << ".dump";
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }
  
  return 0;
}
