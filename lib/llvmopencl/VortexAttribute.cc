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

#include "Workgroup.h"

POP_COMPILER_DIAGS

using namespace llvm;

namespace {
class VortexRemoveAttr : public ModulePass {

  public:
  static char ID;
  VortexRemoveAttr() : ModulePass(ID) {}

  virtual bool runOnModule(Module& M);
};

} // namespace

char VortexRemoveAttr::ID = 0;
static RegisterPass<VortexRemoveAttr>
    X("vortex-mno-riscv-attribute",
        "print module dump");

static void recursivelyFind(Function* F, llvm::Attribute attr)
{

  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    for (BasicBlock::iterator BI = I->begin(), BE = I->end(); BI != BE; ++BI) {

      Instruction* instr = dyn_cast<Instruction>(BI);
      if (!llvm::isa<CallInst>(instr))
        continue;

      CallInst* call_inst = dyn_cast<CallInst>(instr);
      Function* callee = call_inst->getCalledFunction();

      if ((callee == nullptr) || callee->getName().startswith("llvm."))
        continue;
      
      if(callee->hasFnAttribute("target-features")){
        callee->removeFnAttr("target-features");
        callee->addFnAttr(attr.getKindAsString(), attr.getValueAsString());

      }else if(callee->hasFnAttribute("target-cpu")){
        callee->removeFnAttr("target-cpu");
      }

      recursivelyFind(callee, attr);
    }
  }
  return;
}


bool VortexRemoveAttr::runOnModule(Module& M)
{
  std::string KernelName;
  getModuleStringMetadata(M, "KernelName", KernelName);

  if(KernelName == "")
    return false;

  for (llvm::Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    llvm::Function* F = &*I;
    if (F->isDeclaration())
      continue;

    if (KernelName == F->getName().str() || pocl::Workgroup::isKernelToProcess(*F)) {

      /*{
        const llvm::AttributeList& attrlist = F->getAttributes();
        std::string str;
        llvm::raw_string_ostream ostream {str};
        attrlist.print(ostream);
        std::cerr << "### VortexNoRISCVAttribute Pass running on " << KernelName
          << str 
          << std::endl;
      }*/
     
      if(F->hasFnAttribute("target-features")){
        llvm::Attribute target_features = F->getFnAttribute("target-features");
        recursivelyFind(F, target_features); 
      }
       
    }
  }
 
  return true;
}
