#include <iostream>
#include <set>
#include <string>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "config.h"
#include "pocl.h"
#include "pocl_llvm_api.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <llvm/IR/Instructions.h>

#include "Barrier.h"
#include "Workgroup.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace {
class VortexPrintfLowering : public ModulePass {

  public:
  static char ID;
  VortexPrintfLowering() : ModulePass(ID) {}

  virtual bool runOnModule(Module& M);
};

} // namespace

char VortexPrintfLowering::ID = 0;
static RegisterPass<VortexPrintfLowering>
    X("vortex-printfs",
        "Lower printf to vortex printf function");

#define DEBUG_VORTEX_PRINTF

static void printfLowering(Module& M, Instruction* inst, 
    std::vector<Instruction *>& need_remove){

  LLVMContext&context = M.getContext();
  auto call_inst = dyn_cast<CallInst>(inst);

  std::vector<llvm::Type *> args;
  args.push_back(llvm::Type::getInt8PtrTy(context));
  llvm::FunctionType *printfType =
      FunctionType::get(llvm::Type::getInt32Ty(context), args, true);

  llvm::FunctionCallee _f =
      M.getOrInsertFunction("vx_printf", printfType);
  llvm::Function *func_printf =
      llvm::cast<llvm::Function>(_f.getCallee());

  std::vector<Value *> printf_args;
  for(int i = 0; i < call_inst->arg_size(); i++){
    printf_args.push_back(call_inst->getArgOperand(i));
  }

  auto vx_printf_inst = llvm::CallInst::Create(func_printf, printf_args, "", inst);

  need_remove.push_back(inst);
  inst->replaceAllUsesWith(vx_printf_inst);

  return; 
}

static void recursivelyFind(Module& M, Function* F, 
  std::vector<Instruction *>& need_remove)
{

#ifdef DEBUG_VORTEX_PRINTF
  std::cerr << "### VortexPrintfLowering: SCANNING " << F->getName().str()
            << std::endl;
#endif

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {

      Instruction* instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(instr))
        continue;

      CallInst* call_inst = dyn_cast<CallInst>(instr);
      Function* callee = call_inst->getCalledFunction();

      if ((callee == nullptr) || callee->getName().startswith("llvm."))
        continue;

      auto func_name = callee->getName().str();
      if(func_name == "printf"){
        printfLowering(M, instr, need_remove); 
 
#ifdef DEBUG_VORTEX_PRINTF
  std::cerr << "### VortexPrintfLowering: Find " << F->getName().str()
            << std::endl;
#endif
     
      }else {
        recursivelyFind(M, callee, need_remove);
      }
    }
  }
  return;
}

bool VortexPrintfLowering::runOnModule(Module& M)
{
  bool changed = false;
  std::vector<Instruction*> need_remove;

  std::string KernelName;
  getModuleStringMetadata(M, "KernelName", KernelName);

  if(KernelName == "")
    return false;

  // Find Printf recursively
  for (llvm::Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    llvm::Function* F = &*I;
    if (F->isDeclaration())
      continue;

    if (KernelName == F->getName().str() || pocl::Workgroup::isKernelToProcess(*F)) {

#ifdef DEBUG_VORTEX_PRINTF
      std::cerr << "### VortexPrintfLowering Pass running on " << KernelName
                << std::endl;
#endif
      // we don't want to set alwaysInline on a Kernel, only its subroutines.
      recursivelyFind(M, F, need_remove);
    }
  }

  {
    std::string str;
    llvm::raw_string_ostream ostream { str };
    M.print(ostream, nullptr, false);
    std::cout << str << std::endl;
  }

  for(auto inst : need_remove){
    inst->eraseFromParent();
  }

  return changed;
}
