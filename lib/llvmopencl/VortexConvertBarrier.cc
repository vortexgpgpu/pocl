#include <iostream>
#include <set>
#include <string>

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "config.h"
#include "pocl.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include <llvm/IR/Instructions.h>

#include "Barrier.h"
#include "Workgroup.h"

#include "VortexFunctionLink.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace {
class VortexBarrierLowering : public ModulePass {

  public:
  static char ID;
  VortexBarrierLowering()
      : ModulePass(ID)
  {
  }

  virtual bool runOnModule(Module& M);
};

} // namespace

//#define DEBUG_VORTEX_BARRIER

extern cl::opt<std::string> KernelName;

char VortexBarrierLowering::ID = 0;
static RegisterPass<VortexBarrierLowering>
    X("vortex-barrier-lowering",
        "Lower pocl barrier to vortex barrier function");

static void recursivelyConvertBarrier(Function* F, std::set<Instruction*>& Barriers)
{

#ifdef DEBUG_VORTEX_BARRIER
  std::cerr << "### VortexBarrierLowering: SCANNING " << F->getName().str()
            << std::endl;
#endif

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {
      // Pass barrier in First Block, implict barrier
      if (I == F->begin())
        continue;

      Instruction* Instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(Instr))
        continue;

      CallInst* CallInstr = dyn_cast<CallInst>(Instr);
      Function* Callee = CallInstr->getCalledFunction();

      if ((Callee == nullptr) || Callee->getName().startswith("llvm."))
        continue;

      if (llvm::isa<pocl::Barrier>(CallInstr)) {
        // Pass barrier in Exit block
        if (Instr->getNextNode() != nullptr)
          if (llvm::isa<llvm::ReturnInst>(Instr->getNextNode()))
            continue;

#ifdef DEBUG_VORTEX_BARRIER
        std::cerr << "### VortexBarrierLowering: find barrier"
                  << F->getName().str() << std::endl;
#endif
        Barriers.insert(Instr);
      } else {
        recursivelyConvertBarrier(Callee, Barriers);
      }
    }
  }
  return;
}

bool VortexBarrierLowering::runOnModule(Module& M)
{
  bool Changed = false;
  std::set<Instruction*> Barriers;
  std::set<Instruction*> debug;

  // Find barrier
  for (llvm::Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    llvm::Function* F = &*I;
    if (F->isDeclaration())
      continue;

    if (KernelName == F->getName() || pocl::Workgroup::isKernelToProcess(*F)) {

#ifdef DEBUG_VORTEX_BARRIER
      std::cerr << "### VortexBarrierLowering Pass running on " << KernelName
                << std::endl;
#endif
      // we don't want to set alwaysInline on a Kernel, only its subroutines.
      recursivelyConvertBarrier(F, Barriers);
    }
  }

  if (!Barriers.empty()) {
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
    for (auto B : Barriers) {
      IRBuilder<> builder(B);
      CallInst* nW = builder.CreateCall(nWC);
      auto curBNum = llvm::ConstantInt::get(
          context, llvm::APInt(32, curBNum_++, false));
      CallInst* vxbar = builder.CreateCall(VXBarF, { curBNum, nW });
    }

    Changed = true;
  }
  return Changed;
}
