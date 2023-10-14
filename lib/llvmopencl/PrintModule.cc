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


POP_COMPILER_DIAGS

using namespace llvm;

namespace {
class PrintModule : public ModulePass {

  public:
  static char ID;
  PrintModule() : ModulePass(ID) {}

  virtual bool runOnModule(Module& M);
};

} // namespace

char PrintModule::ID = 0;
static RegisterPass<PrintModule>
    X("print-module",
        "print module dump");

bool PrintModule::runOnModule(Module& M)
{
  {
    std::string str;
    llvm::raw_string_ostream ostream { str };
    M.print(ostream, nullptr, false);


    std::cout << "\n\n============================\n Print Module\n"
    << str << std::endl;
  }
  return false;
}
