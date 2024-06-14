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

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>

#if LLVM_MAJOR >= 17
#include <llvm/Transforms/IPO/Internalize.h>
#endif

#include "vortex_utils.h"

#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Bitcode/BitcodeWriter.h>

#include "pocl.h"
#include "pocl_file_util.h"
#include "pocl_util.h"

#include "LLVMUtils.h"
#include "kernel_args.h"

static int exec(const char* cmd, std::ostream& out) {
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

void remove_extension(char* filename) {
  char *last_dot = strrchr(filename, '.');
  if (last_dot != NULL) {
    *last_dot = '\0';
  }
}

static char* convertToCharArray(const llvm::SmallVector<std::string, 8>& names) {
  // Calculate the total length required for the buffer
  size_t totalLength = 0;
  for (const auto& name : names) {
    totalLength += name.size() + 1; // +1 for the null terminator
  }

  // Allocate buffer
  char* buffer = (char*)malloc(totalLength * sizeof(char));
  if (buffer == nullptr) {
    std::cerr << "Memory allocation failed" << std::endl;
    return nullptr;
  }

  // Copy names into buffer with null separation
  size_t position = 0;
  for (const auto& name : names) {
    std::strcpy(buffer + position, name.c_str());
    position += name.size();
    buffer[position] = '\0'; // Null terminator
    position += 1;
  }

  return buffer;
}

// Store function arguments in a single argument buffer.
static bool createArgumentsBuffer(llvm::Function *function, llvm::Module *module, llvm::SmallVector<std::string, 8>& funcNames) {
  auto &Context = module->getContext();
  const llvm::DataLayout &DL = module->getDataLayout();

  std::string TargetTriple = module->getTargetTriple();
  bool is64Bit = TargetTriple.find("riscv64") != std::string::npos;

  auto I32Ty = llvm::Type::getInt32Ty(Context);
  auto I8Ty = llvm::Type::getInt8Ty(Context);
  auto I8PtrTy = I8Ty->getPointerTo();

  // Create new function signature
  auto NewFuncType = llvm::FunctionType::get(function->getReturnType(), {I8PtrTy}, false);
  auto NewFunc = llvm::Function::Create(NewFuncType, function->getLinkage(), function->getName() + "_vortex");
  module->getFunctionList().insert(function->getIterator(), NewFunc);
  NewFunc->takeName(function);

  auto EntryBlock = llvm::BasicBlock::Create(Context, "entry", NewFunc);
  llvm::IRBuilder<> Builder(EntryBlock);

  // Access function arguments
  auto ai = NewFunc->arg_begin();
  auto ArgBuffer = &*ai++;
  ArgBuffer->setName("ArgBuffer");

  unsigned arg_idx = 0;
  unsigned arg_offset = 0;

  llvm::Value* allocated_local_mem = nullptr;

  auto MDS = llvm::MDNode::get(Context, llvm::MDString::get(Context, "vortex.uniform"));

  uint32_t BaseAlignment = is64Bit ? 8 : 4;

  for (auto& OldArg : function->args()) {
    auto ArgType = OldArg.getType();
    auto ArgOffset = llvm::ConstantInt::get(I32Ty, arg_offset);
    llvm::Value* Arg;
    if (pocl::isLocalMemFunctionArg(function, arg_idx)) {
      if (allocated_local_mem == nullptr) {
        // Load __local_size
        auto local_size_ptr = Builder.CreateGEP(I8Ty, ArgBuffer, ArgOffset, "__local_size_ptr");
        auto local_size = Builder.CreateLoad(I32Ty, local_size_ptr, "__local_size");
        arg_offset = alignOffset(arg_offset + 4, BaseAlignment);
        // Call vx_local_alloc(__local_size)
        auto function_type = llvm::FunctionType::get(I8PtrTy, {I32Ty}, false);
        auto vx_local_alloc_func = module->getOrInsertFunction("vx_local_alloc", function_type);
        allocated_local_mem = Builder.CreateCall(vx_local_alloc_func, {local_size}, "__local_mem");
      }
      // Load argument __offset
      auto offset_ptr = Builder.CreateGEP(I8Ty, ArgBuffer, ArgOffset, OldArg.getName() + "_offset_ptr");
      auto offset = Builder.CreateLoad(I32Ty, offset_ptr, OldArg.getName() + "_offset");
      arg_offset = alignOffset(arg_offset + 4, BaseAlignment);
      // Apply pointer offset
      Arg = Builder.CreateGEP(I8PtrTy, allocated_local_mem, offset, OldArg.getName() + "_byte_ptr");
    } else {
      auto offset_ptr = Builder.CreateGEP(I8Ty, ArgBuffer, ArgOffset, OldArg.getName() + "_offset_ptr");
      Arg = Builder.CreateLoad(ArgType, offset_ptr, OldArg.getName() + "_loaded");
      arg_offset = alignOffset(arg_offset + DL.getTypeAllocSize(ArgType), BaseAlignment);
    }
    auto instr = llvm::cast<llvm::Instruction>(Arg);
    assert(instr != nullptr);
    instr->setMetadata("vortex.uniform", MDS);

    OldArg.replaceAllUsesWith(Arg);
    arg_idx += 1;
  }

  // Move the body of the old function to the new function
  NewFunc->splice(NewFunc->end(), function);

  // Connect the entry block to the first block of the old function
  for (auto& BB : *NewFunc) {
    if (&BB != EntryBlock) {
      Builder.CreateBr(&BB);
      break;
    }
  }

  funcNames.push_back(NewFunc->getName().str());

  return true;
}

static void processKernels(llvm::SmallVector<std::string, 8>& funcNames, llvm::Module *module) {
  llvm::SmallVector<llvm::Function *, 8> functionsToErase;
  for (auto& function : module->functions()) {
    if (!pocl::isKernelToProcess(function))
      continue;
    if (createArgumentsBuffer(&function, module, funcNames))
      functionsToErase.push_back(&function);
  }
  for (auto function : functionsToErase) {
    function->eraseFromParent();
  }
}

static void addKernelSelect(llvm::SmallVector<std::string, 8>& funcNames, llvm::Module *module) {
  auto& Context = module->getContext();

  auto I32Ty = llvm::Type::getInt32Ty(Context);
  auto I8Ty = llvm::Type::getInt8Ty(Context);
  auto I8PtrTy = I8Ty->getPointerTo();
  auto GetKernelCallbackTy = llvm::FunctionType::get(I8PtrTy, {I32Ty}, false);

  auto GetKernelCallbackFunc = llvm::Function::Create(
    GetKernelCallbackTy, llvm::Function::ExternalLinkage, "__vx_get_kernel_callback", module);

  llvm::IRBuilder<> Builder(Context);

  auto EntryBB = llvm::BasicBlock::Create(Context, "entry", GetKernelCallbackFunc);
  Builder.SetInsertPoint(EntryBB);

  // Get the function argument (kernel_index)
  auto Args = GetKernelCallbackFunc->arg_begin();
  auto KernelIndex = Args++;
  KernelIndex->setName("kernel_index");

  // Prepare the switch instruction
  auto Switch = Builder.CreateSwitch(KernelIndex, EntryBB);

  // Iterate through the functions in the module and create cases for the switch
  int FunctionIndex = 0;
  for (llvm::Function& F : module->functions()) {
    if (std::find(funcNames.begin(), funcNames.end(), F.getName().str()) == funcNames.end())
      continue;
    // Create a basic block for this function index
    auto CaseBB = llvm::BasicBlock::Create(Context, "case_" + std::to_string(FunctionIndex), GetKernelCallbackFunc);
    Builder.SetInsertPoint(CaseBB);
    // Return the function pointer
    Builder.CreateRet(Builder.CreateBitCast(&F, GetKernelCallbackTy->getReturnType()));
    // Add the case to the switch statement
    Switch->addCase(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), FunctionIndex), CaseBB);
    ++FunctionIndex;
  }
  // Create a default case block for out-of-bounds indices
  auto DefaultBB = llvm::BasicBlock::Create(Context, "default", GetKernelCallbackFunc);
  Builder.SetInsertPoint(DefaultBB);
  Builder.CreateRet(llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(GetKernelCallbackTy->getReturnType())));
  Switch->setDefaultDest(DefaultBB);
}

int compile_vortex_program(char**kernel_names, int* num_kernels, char* sz_program_vxbin, void* llvm_module) {
  int err;

  const char* llvm_install_path = getenv("LLVM_PREFIX");
  if (llvm_install_path) {
    if (!pocl_exists(llvm_install_path)) {
      POCL_MSG_ERR("$LLVM_PREFIX: '%s' doesn't exist\n", llvm_install_path);
      return -1;
    }
    POCL_MSG_PRINT_INFO("using $LLVM_PREFIX=%s!\n", llvm_install_path);
  }

  std::string build_cflags = pocl_get_string_option("POCL_VORTEX_CFLAGS", "");
  if (build_cflags == "") {
    POCL_MSG_ERR("'POCL_VORTEX_CFLAGS' need to be set\n");
    return -1;
  }

  std::string build_ldflags = pocl_get_string_option ("POCL_VORTEX_LDFLAGS", "");
  if(build_ldflags == ""){
    POCL_MSG_ERR("'POCL_VORTEX_LDFLAGS' need to be set\n");
    return -1;
  }

  char sz_program_bc[POCL_MAX_PATHNAME_LENGTH + 1];
  err = pocl_mk_tempname(sz_program_bc, "/tmp/pocl_vortex_program", ".bc", nullptr);
  if (err != 0)
    return err;

  char sz_program_elf[POCL_MAX_PATHNAME_LENGTH + 1];
  err = pocl_mk_tempname(sz_program_elf, "/tmp/pocl_vortex_program", ".elf", nullptr);
  if (err != 0)
    return err;

  auto module = (llvm::Module *)llvm_module;
  llvm::SmallVector<std::string, 8> kernelNames;
  processKernels(kernelNames, module);
  addKernelSelect(kernelNames, module);

  *num_kernels = kernelNames.size();
  *kernel_names = convertToCharArray(kernelNames);

  {
    std::error_code EC;
    llvm::raw_fd_ostream file(sz_program_bc, EC, llvm::sys::fs::OF_None);
    llvm::WriteBitcodeToFile(*module, file);
    file.close();
  }

  if (POCL_DEBUGGING_ON) {
    std::error_code EC;
    llvm::raw_fd_ostream file("program.ll", EC, llvm::sys::fs::OF_None);
    module->print(file, nullptr);
    file.close();
  }

  {
    std::string clang_path(CLANG);
    if (llvm_install_path) {
      clang_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path);
    }

    char sz_kernel_main[POCL_MAX_PATHNAME_LENGTH];
    pocl_get_srcdir_or_datadir (sz_kernel_main, "/lib/CL/devices", "", "/vortex/kernel_main.c");

    std::stringstream ss_cmd, ss_out;
    ss_cmd << clang_path.c_str() << " " << build_cflags << " " << sz_program_bc << " " << sz_kernel_main << " " << build_ldflags << " -o " << sz_program_elf;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  if (POCL_DEBUGGING_ON) {
    std::string objdump_path(LLVM_OBJDUMP);
    if (llvm_install_path) {
      objdump_path.replace(0, strlen(LLVM_PREFIX), llvm_install_path);
    }

    std::stringstream ss_cmd, ss_out;
    ss_cmd << objdump_path.c_str() << " -D " << sz_program_elf << " > program.dump";

    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  {
    std::string vxbintool_path = pocl_get_string_option ("POCL_VORTEX_BINTOOL", "");
    if (vxbintool_path == ""){
      POCL_MSG_ERR("'POCL_VORTEX_BINTOOL' need to be set\n");
      return -1;
    }
    std::stringstream ss_cmd, ss_out;
    ss_cmd << vxbintool_path << " " << sz_program_elf << " " << sz_program_vxbin;
    POCL_MSG_PRINT_LLVM("running \"%s\"\n", ss_cmd.str().c_str());
    int err = exec(ss_cmd.str().c_str(), ss_out);
    if (err != 0) {
      POCL_MSG_ERR("%s\n", ss_out.str().c_str());
      return err;
    }
  }

  return 0;
}