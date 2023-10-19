#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "pocl.h"

#include "CompilerWarnings.h"
IGNORE_COMPILER_WARNING("-Wunused-parameter")

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "workitem-loops-vx"

#include "WorkitemLoops.h"
#include "Workgroup.h"
#include "Barrier.h"
#include "Kernel.h"
#include "WorkitemHandlerChooser.h"


//#define DUMP_CFGS

#include "DebugHelpers.h"

//#define DEBUG_WORK_ITEM_LOOPS

#include "VariableUniformityAnalysis.h"

using namespace llvm;
using namespace pocl;

void printIR(llvm::Function* F_)
{
  std::string str;
  llvm::raw_string_ostream ostream { str };
  F_->print(ostream, nullptr, false);
  std::cout << str << std::endl;
}

void WorkitemLoops::CreateVortexVar(
    llvm::Function* F, VortexData& tmdata, int schedule)
{
  IRBuilder<> builder(&*(F->getEntryBlock().getFirstInsertionPt()));
  auto M = F->getParent();

  // Load Global variable for Vortex lowering
  GlobalVariable* nLx = M->getGlobalVariable("_local_size_x");
  if (nLx == NULL)
    nLx = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
        NULL, "_local_size_x", NULL,
        GlobalValue::ThreadLocalMode::NotThreadLocal,
        0, true);

  GlobalVariable* nLy = M->getGlobalVariable("_local_size_y");
  if (nLy == NULL)
    nLy = new GlobalVariable(*M, SizeT, false, GlobalValue::CommonLinkage,
        NULL, "_local_size_y");

  GlobalVariable* nLz = M->getGlobalVariable("_local_size_z");
  if (nLz == NULL)
    nLz = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
        NULL, "_local_size_z", NULL,
        GlobalValue::ThreadLocalMode::NotThreadLocal,
        0, true);

  LLVMContext& context = M->getContext();
  auto inty = IntegerType::get(context, SizeTWidth);

  LoadInst* loadLx = builder.CreateLoad(inty, nLx, "nl_x");
  LoadInst* loadLy = builder.CreateLoad(inty, nLy, "nl_y");
  LoadInst* loadLz = builder.CreateLoad(inty, nLz, "nl_z");
  auto loadLxy = builder.CreateBinOp(Instruction::Mul, loadLx, loadLy, "nl_xy");
  auto loadLxyz = builder.CreateBinOp(Instruction::Mul, loadLxy, loadLz, "nl_xyz");

  // Generate function def for getting VX function
  //FunctionType* nTTy = FunctionType::get(IntegerType::getInt32Ty(context), true);
  FunctionType* nTTy = FunctionType::get(inty, true);

  FunctionCallee tidC = M->getOrInsertFunction("vx_thread_id", nTTy);
  FunctionCallee widC = M->getOrInsertFunction("vx_warp_id", nTTy);
  FunctionCallee nHTC = M->getOrInsertFunction("vx_num_threads", nTTy);
  FunctionCallee nHWC = M->getOrInsertFunction("vx_num_warps", nTTy);

  auto tid = builder.CreateCall(tidC, {}, "tid");
  auto wid = builder.CreateCall(widC, {}, "wid");
  auto nHT = builder.CreateCall(nHTC, {}, "nHT");
  auto nHW = builder.CreateCall(nHWC, {}, "nHW");

  auto tlid = builder.CreateAdd(tid, builder.CreateMul(wid, nHW), "tlid");
  auto TpC = builder.CreateBinOp(Instruction::Mul, nHT, nHW, "HTpC");
  auto localIDHolder = builder.CreateAlloca(inty, 0, ".pocl.vortex_local_id");


  tmdata.LocalID = tlid;
  tmdata.TpC = TpC;
  tmdata.workload = loadLxyz;
  tmdata.num_local_x = loadLx;
  tmdata.num_local_xy = loadLxy;
  tmdata.localIDHolder = localIDHolder;

  return;
}

  std::pair<llvm::BasicBlock*, llvm::BasicBlock*>
WorkitemLoops::CreateVortexCMLoop(ParallelRegion& region,
    llvm::BasicBlock* entryBB, llvm::BasicBlock* exitBB,
    VortexData tmdata)
{
  auto lid = tmdata.LocalID;
  auto TpC = tmdata.TpC;
  auto workload = tmdata.workload;
  auto num_local_x = tmdata.num_local_x;
  auto num_local_xy = tmdata.num_local_xy;
  auto localIDHolder = tmdata.localIDHolder;

  llvm::BasicBlock* loopBodyEntryBB = entryBB;
  llvm::LLVMContext& ctx = loopBodyEntryBB->getContext();
  llvm::Function* F = loopBodyEntryBB->getParent();
  auto M = F->getParent();
  LLVMContext& context = M->getContext();
  auto inty = IntegerType::get(context, SizeTWidth);

  loopBodyEntryBB->setName(std::string("pregion_for_entry.") + entryBB->getName().str());
  assert(exitBB->getTerminator()->getNumSuccessors() == 1);

  // Generate basic block
  llvm::BasicBlock* oldExit = exitBB->getTerminator()->getSuccessor(0);
  llvm::BasicBlock* forInitBB = BasicBlock::Create(ctx, "pregion_for_init", F, loopBodyEntryBB);
  llvm::BasicBlock* forIncBB = BasicBlock::Create(ctx, "pregion_for_inc", F, exitBB);
  llvm::BasicBlock* forCondBB = BasicBlock::Create(ctx, "pregion_for_cond", F, exitBB);
  llvm::BasicBlock* loopEndBB = BasicBlock::Create(ctx, "pregion_for_end", F, exitBB);

  // Run DominatorTree
  DTP->runOnFunction(*F);

  /* Collect the basic blocks in the parallel region that dominate the
     exit. These are used in determining whether load instructions may
     be executed unconditionally in the parallel loop (see below). */
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> dominatesExitBB;
  for (auto bb: region) {
    if (DT->dominates(bb, exitBB)) {
      dominatesExitBB.insert(bb);
    }
  }

  // Replace Terminator
  BasicBlockVector preds;
  llvm::pred_iterator PI = llvm::pred_begin(entryBB),
    E = llvm::pred_end(entryBB);

  for (; PI != E; ++PI) {
    llvm::BasicBlock* bb = *PI;
    preds.push_back(bb);
  }

  for (BasicBlockVector::iterator it = preds.begin();
      it != preds.end(); ++it) {
    llvm::BasicBlock* bb = *it;
    if (DT->dominates(loopBodyEntryBB, bb))
      continue;
    bb->getTerminator()->replaceUsesOfWith(loopBodyEntryBB, forInitBB);
  }

  // Add inst for InitBlock, jump to condition block
  IRBuilder<> builder(forInitBB);
  builder.CreateStore(lid, localIDHolder);
  builder.CreateBr(forCondBB);

  exitBB->getTerminator()->replaceUsesOfWith(oldExit, forCondBB);

  // Add instruction for Inc Block
  llvm::BasicBlock* oldEntry = exitBB->getTerminator()->getSuccessor(0);
  assert(oldEntry != NULL);
  exitBB->getTerminator()->replaceUsesOfWith(oldEntry, forIncBB);

  builder.SetInsertPoint(forIncBB);
  auto add = builder.CreateAdd(builder.CreateLoad(inty, localIDHolder), TpC);
  builder.CreateStore(add, localIDHolder);
  builder.CreateBr(oldEntry);

  // Add instruction for Cond Block
  builder.SetInsertPoint(forCondBB);
  auto curid = builder.CreateLoad(inty, localIDHolder);
  llvm::Value* cmpResult = builder.CreateICmpULT(curid, workload);
  Instruction* loopBranch = builder.CreateCondBr(
      cmpResult, loopBodyEntryBB, loopEndBB);

  // Replace usage of local id to vortex drived local id
  {
    bool flag_x = false, flag_y = false, flag_z = false;

    for (auto I = region.begin(); I != region.end(); ++I) {
      llvm::BasicBlock* bb = *I;
      if (!DT->dominates(loopBodyEntryBB, bb))
        continue;

      for (BasicBlock::iterator BI = bb->begin(); BI != bb->end(); ++BI) {
        Instruction* Instr = dyn_cast<Instruction>(BI);
        if(Instr->use_empty())
          continue;

        if (isa<llvm::LoadInst>(Instr)) {
          llvm::LoadInst* load = dyn_cast<llvm::LoadInst>(Instr);
          llvm::Value* pointer = load->getPointerOperand();
          StringRef pname = pointer->getName();

          if (pname.equals("_local_id_x")) {
            flag_x = true;
          } else if (pname.equals("_local_id_y")) {
            flag_y = true;
          } else if (pname.equals("_local_id_z")) {
            flag_z = true;
          }
        }
      }
    }

    builder.SetInsertPoint(&(loopBodyEntryBB->front()));
    auto curid = builder.CreateLoad(inty, localIDHolder);

    // simple version
    /*
    auto local_id_z = builder.CreateUDiv(curid, num_local_xy, "new_lid_z");
    auto remains_xy = builder.CreateSub(curid,
        builder.CreateMul(local_id_z, num_local_xy));
    auto local_id_y = builder.CreateUDiv(remains_xy, num_local_x, "new_lid_y");
    auto local_id_x = builder.CreateSub(remains_xy,
        builder.CreateMul(local_id_y, num_local_x), "new_lid_x");
    */
    llvm::Value *local_id_x, *local_id_y, *local_id_z;

    if(flag_x & !flag_y && !flag_z){
      local_id_x = builder.CreateURem(curid, num_local_x, "new_lid_x");
    }else {
      local_id_z = builder.CreateUDiv(curid, num_local_xy, "new_lid_z");
      auto remains_xy = builder.CreateSub(curid,
        builder.CreateMul(local_id_z, num_local_xy));
      local_id_y = builder.CreateUDiv(remains_xy, num_local_x, "new_lid_y");
      local_id_x = builder.CreateSub(remains_xy,
        builder.CreateMul(local_id_y, num_local_x), "new_lid_x");
    }
  
    auto M = F->getParent();
    std::vector<Instruction*> trashs;

    for (auto I = region.begin(); I != region.end(); ++I) {
      llvm::BasicBlock* bb = *I;
      //std::cerr << "### VortexLoopGen : pred Name : " << bb->getName().str()
      //  << std::endl;

      if (!DT->dominates(loopBodyEntryBB, bb))
        continue;

      for (BasicBlock::iterator BI = bb->begin(); BI != bb->end(); ++BI) {
        Instruction* Instr = dyn_cast<Instruction>(BI);

        if (isa<llvm::LoadInst>(Instr)) {
          llvm::LoadInst* load = dyn_cast<llvm::LoadInst>(Instr);
          llvm::Value* pointer = load->getPointerOperand();
          StringRef pname = pointer->getName();

          //std::cerr << "### VortexLoopGen : " << pname.str()
          //  << " : " << pname.equals("_local_id_x") << std::endl;

          if (pname.equals("_local_id_x")) {
            Instr->replaceAllUsesWith(local_id_x);
            trashs.push_back(Instr);
          } else if (flag_y && pname.equals("_local_id_y")) {
            Instr->replaceAllUsesWith(local_id_y);
            trashs.push_back(Instr);
          } else if (flag_z && pname.equals("_local_id_z")) {
            Instr->replaceAllUsesWith(local_id_z);
            trashs.push_back(Instr);
          }
        }
      }
    }
  std::cerr << "### VortexLoopGen : before trash" << std::endl;
    for (auto BI : trashs) {
      BI->eraseFromParent();
    }
  }

  std::cerr << "### VortexLoopGen : Finish structure gen" << std::endl;

  /* Add the metadata to mark a parallel loop. The metadata 
     refer to a loop-unique dummy metadata that is not merged
     automatically. */

  /* This creation of the identifier metadata is copied from
     LLVM's MDBuilder::createAnonymousTBAARoot(). */

  MDNode *Dummy = MDNode::getTemporary(context, ArrayRef<Metadata*>()).release();
  MDNode *AccessGroupMD = MDNode::getDistinct(context, {});
  MDNode *ParallelAccessMD = MDNode::get(
      context, {MDString::get(context, "llvm.loop.parallel_accesses"), AccessGroupMD});

  MDNode *Root = MDNode::get(context, {Dummy, ParallelAccessMD});

  // At this point we have
  //   !0 = metadata !{}            <- dummy
  //   !1 = metadata !{metadata !0} <- root
  // Replace the dummy operand with the root node itself and delete the dummy.
  Root->replaceOperandWith(0, Root);
  MDNode::deleteTemporary(Dummy);
  // We now have
  //   !1 = metadata !{metadata !1} <- self-referential root
  loopBranch->setMetadata("llvm.loop", Root);

  auto IsLoadUnconditionallySafe =
    [&dominatesExitBB](llvm::Instruction *insn) -> bool {
      assert(insn->mayReadFromMemory());
      // Checks that the instruction isn't in a conditional block.
      return dominatesExitBB.count(insn->getParent());
    };

  region.AddParallelLoopMetadata(AccessGroupMD, IsLoadUnconditionallySafe);


  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  std::cerr << "### VortexLoopGen : End program" << std::endl;
  //printIR(F);

  return std::make_pair(forInitBB, loopEndBB);
}
