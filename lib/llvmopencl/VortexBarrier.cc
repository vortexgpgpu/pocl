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
class VortexBarrierLowering : public ModulePass {

  public:
  static char ID;
  VortexBarrierLowering() : ModulePass(ID) {}

  virtual bool runOnModule(Module& M);
};

} // namespace

char VortexBarrierLowering::ID = 0;
static RegisterPass<VortexBarrierLowering>
    X("vortex-barriers",
        "Lower pocl barrier to vortex barrier function");

//#define DEBUG_VORTEX_CONVERT


static void recursivelyFind(Function* F, std::set<Instruction*>& barriers)
{

#ifdef DEBUG_VORTEX_CONVERT
  std::cerr << "### VortexBarrierLowering: SCANNING " << F->getName().str()
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

      if (llvm::isa<pocl::Barrier>(call_inst)) {
        // Pass barrier in First Block, implict barrier
        if (I == F->begin())
          continue;

        // Pass barrier in Exit block
        if (instr->getNextNode() != nullptr)
          if (llvm::isa<llvm::ReturnInst>(instr->getNextNode()))
            continue;

        barriers.insert(instr);

      }else {
        recursivelyFind(callee, barriers);
      }
    }
  }
  return;
}

bool VortexBarrierLowering::runOnModule(Module& M)
{

  int vortex_scheduling_flag = 0;  
  if(std::getenv("VORTEX_SCHEDULE_FLAG") != nullptr)
    std::stoi(std::string(std::getenv("VORTEX_SCHEDULE_FLAG")));
  if(vortex_scheduling_flag == 0)
    return false;

  std::set<Instruction*> barriers;

  std::string KernelName;
  getModuleStringMetadata(M, "KernelName", KernelName);

  if(KernelName == "")
    return false;

  // Find barrier
  for (llvm::Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    llvm::Function* F = &*I;
    if (F->isDeclaration())
      continue;

    if (KernelName == F->getName().str() || pocl::Workgroup::isKernelToProcess(*F)) {

#ifdef DEBUG_VORTEX_CONVERT
      std::cerr << "### VortexBarrierLowering Pass running on " << KernelName
                << std::endl;
#endif
      // we don't want to set alwaysInline on a Kernel, only its subroutines.
      recursivelyFind(F, barriers);
    }
  }

  if (!barriers.empty()) {
    LLVMContext& context = M.getContext();
    // Generate function def for getting VX Warp Size
    FunctionType* nTTy = FunctionType::get(IntegerType::getInt32Ty(context), true);
    FunctionCallee nWC = M.getOrInsertFunction("vx_num_warps", nTTy);

    // Generate function def for VX Barrier
    ArrayRef<Type*> VXBParams = { IntegerType::getInt32Ty(context), IntegerType::getInt32Ty(context) };
    FunctionType* VXBarTy = FunctionType::get(Type::getVoidTy(context), VXBParams, true);
    FunctionCallee VXBarC = M.getOrInsertFunction("vx_barrier", VXBarTy);
    Function* VXBarF = dyn_cast<Function>(VXBarC.getCallee());

    int curBNum_ = 1;
    for (auto B : barriers) {
      IRBuilder<> builder(B);
      CallInst* nW = builder.CreateCall(nWC);
      auto curBNum = llvm::ConstantInt::get(
          context, llvm::APInt(32, curBNum_++, false));
      CallInst* vxbar = builder.CreateCall(VXBarF, { curBNum, nW });
    }
    return true;
  }

  return false;
}
