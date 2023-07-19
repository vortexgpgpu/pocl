// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define DEBUG_TYPE "workitem-loops"

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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "Barrier.h"
#include "Kernel.h"
#include "Workgroup.h"
#include "WorkitemHandlerChooser.h"
#include "WorkitemLoops.h"

//#define DUMP_CFGS

#include "DebugHelpers.h"

//#define DEBUG_WORK_ITEM_LOOPS

#include "VariableUniformityAnalysis.h"

#define CONTEXT_ARRAY_ALIGN 64

using namespace llvm;
using namespace pocl;

void printIR(llvm::Function* F_)
{
  std::string str;
  llvm::raw_string_ostream ostream { str };
  F_->print(ostream, nullptr, false);
  std::cout << str << std::endl;
}

namespace {
  static RegisterPass<WorkitemLoops> X("workitemloops",
      "Workitem loop generation pass");
}

char WorkitemLoops::ID = 0;

void WorkitemLoops::getAnalysisUsage(AnalysisUsage& AU) const
{
  AU.addRequired<PostDominatorTreeWrapperPass>();

  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();

  AU.addRequired<VariableUniformityAnalysis>();
  AU.addPreserved<pocl::VariableUniformityAnalysis>();

  AU.addRequired<pocl::WorkitemHandlerChooser>();
  AU.addPreserved<pocl::WorkitemHandlerChooser>();
}

bool WorkitemLoops::runOnFunction(Function& F)
{
  if (!Workgroup::isKernelToProcess(F))
    return false;

  if (getAnalysis<pocl::WorkitemHandlerChooser>().chosenHandler() != pocl::WorkitemHandlerChooser::POCL_WIH_LOOPS)
    return false;

  DTP = &getAnalysis<DominatorTreeWrapperPass>();
  DT = &DTP->getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>();

  PDT = &getAnalysis<PostDominatorTreeWrapperPass>();

  tempInstructionIndex = 0;

  //  F.viewCFGOnly();

  bool changed = ProcessFunction(F);

#ifdef DUMP_CFGS
  dumpCFG(F, F.getName().str() + "_after_wiloops.dot",
      original_parallel_regions);
#endif

#if 0
  std::cerr << "### after:" << std::endl;
  F.viewCFG();
#endif

  changed |= fixUndominatedVariableUses(DTP, F);

#if 0
  /* Split large BBs so we can print the Dot without it crashing. */
  changed |= chopBBs(F, *this);
  F.viewCFG();
#endif
  contextArrays.clear();
  tempInstructionIds.clear();

  releaseParallelRegions();

  return changed;
}

  std::pair<llvm::BasicBlock*, llvm::BasicBlock*>
WorkitemLoops::CreateLoopAround(ParallelRegion& region,
    llvm::BasicBlock* entryBB, llvm::BasicBlock* exitBB,
    bool peeledFirst, llvm::Value* localIdVar, size_t LocalSizeForDim,
    bool addIncBlock, llvm::Value* DynamicLocalSize)
{
  assert(localIdVar != NULL);

  /*

     Generate a structure like this for each loop level (x,y,z):

     for.init:

     ; if peeledFirst is false:
     store i32 0, i32* %_local_id_x, align 4

     ; if peeledFirst is true (assume the 0,0,0 iteration has been executed earlier)
     ; assume _local_id_x_first is is initialized to 1 in the peeled pregion copy
     store _local_id_x_first, i32* %_local_id_x, align 4
     store i32 0, %_local_id_x_first

     br label %for.body

     for.body:

     ; the parallel region code here

     br label %for.inc

     for.inc:

     ; Separated inc and cond check blocks for easier loop unrolling later on.
     ; Can then chain N times for.body+for.inc to unroll.

     %2 = load i32* %_local_id_x, align 4
     %inc = add nsw i32 %2, 1

     store i32 %inc, i32* %_local_id_x, align 4
     br label %for.cond

     for.cond:

     ; loop header, compare the id to the local size
     %0 = load i32* %_local_id_x, align 4
     %cmp = icmp ult i32 %0, i32 123
     br i1 %cmp, label %for.body, label %for.end

     for.end:

OPTIMIZE: Use a separate iteration variable across all the loops to iterate the context
data arrays to avoid needing multiplications to find the correct location, and to
enable easy vectorization of loading the context data when there are parallel iterations.
*/

  llvm::BasicBlock* loopBodyEntryBB = entryBB;
  llvm::LLVMContext& C = loopBodyEntryBB->getContext();
  llvm::Function* F = loopBodyEntryBB->getParent();
  loopBodyEntryBB->setName(std::string("pregion_for_entry.") + entryBB->getName().str());

  assert(exitBB->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock* oldExit = exitBB->getTerminator()->getSuccessor(0);

  llvm::BasicBlock* forInitBB = BasicBlock::Create(C, "pregion_for_init", F, loopBodyEntryBB);

  llvm::BasicBlock* loopEndBB = BasicBlock::Create(C, "pregion_for_end", F, exitBB);

  llvm::BasicBlock* forCondBB = BasicBlock::Create(C, "pregion_for_cond", F, exitBB);

  DTP->runOnFunction(*F);

  //  F->viewCFG();
  /* Fix the old edges jumping to the region to jump to the basic block
     that starts the created loop. Back edges should still point to the
     old basic block so we preserve the old loops. */
  BasicBlockVector preds;
  llvm::pred_iterator PI = llvm::pred_begin(entryBB),
    E = llvm::pred_end(entryBB);

  for (; PI != E; ++PI) {
    llvm::BasicBlock* bb = *PI;
    preds.push_back(bb);
  }

  for (BasicBlockVector::iterator i = preds.begin();
      i != preds.end(); ++i) {
    llvm::BasicBlock* bb = *i;
    /* Do not fix loop edges inside the region. The loop
       is replicated as a whole to the body of the wi-loop.*/
    if (DT->dominates(loopBodyEntryBB, bb))
      continue;
    bb->getTerminator()->replaceUsesOfWith(loopBodyEntryBB, forInitBB);
  }

  IRBuilder<> builder(forInitBB);

  if (peeledFirst) {
    builder.CreateStore(builder.CreateLoad(localIdXFirstVar), localIdVar);
    builder.CreateStore(ConstantInt::get(SizeT, 0), localIdXFirstVar);

    if (WGDynamicLocalSize) {
      llvm::Value* cmpResult;
      cmpResult = builder.CreateICmpULT(builder.CreateLoad(localIdVar),
          builder.CreateLoad(DynamicLocalSize));

      builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);
    } else {
      builder.CreateBr(loopBodyEntryBB);
    }
  } else {
    builder.CreateStore(ConstantInt::get(SizeT, 0), localIdVar);

    builder.CreateBr(loopBodyEntryBB);
  }

  exitBB->getTerminator()->replaceUsesOfWith(oldExit, forCondBB);
  if (addIncBlock) {
    AppendIncBlock(exitBB, localIdVar);
  }

  builder.SetInsertPoint(forCondBB);

  llvm::Value* cmpResult;
  if (!WGDynamicLocalSize)
    cmpResult = builder.CreateICmpULT(
        builder.CreateLoad(localIdVar),
        ConstantInt::get(SizeT, LocalSizeForDim));
  else
    cmpResult = builder.CreateICmpULT(
        builder.CreateLoad(localIdVar),
        builder.CreateLoad(DynamicLocalSize));

  Instruction* loopBranch = builder.CreateCondBr(cmpResult, loopBodyEntryBB, loopEndBB);

  /* Add the metadata to mark a parallel loop. The metadata
     refer to a loop-unique dummy metadata that is not merged
     automatically. */

  /* This creation of the identifier metadata is copied from
     LLVM's MDBuilder::createAnonymousTBAARoot(). */

  MDNode* Dummy = MDNode::getTemporary(C, ArrayRef<Metadata*>()).release();
#ifdef LLVM_OLDER_THAN_8_0
  MDNode* Root = MDNode::get(C, Dummy);
#else
  MDNode* AccessGroupMD = MDNode::getDistinct(C, {});
  MDNode* ParallelAccessMD = MDNode::get(
      C, { MDString::get(C, "llvm.loop.parallel_accesses"), AccessGroupMD });

  MDNode* Root = MDNode::get(C, { Dummy, ParallelAccessMD });
#endif

  // At this point we have
  //   !0 = metadata !{}            <- dummy
  //   !1 = metadata !{metadata !0} <- root
  // Replace the dummy operand with the root node itself and delete the dummy.
  Root->replaceOperandWith(0, Root);
  MDNode::deleteTemporary(Dummy);
  // We now have
  //   !1 = metadata !{metadata !1} <- self-referential root
  loopBranch->setMetadata("llvm.loop", Root);

#ifdef LLVM_OLDER_THAN_8_0
  region.AddParallelLoopMetadata(Root);
#else
  region.AddParallelLoopMetadata(AccessGroupMD);
#endif

  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  return std::make_pair(forInitBB, loopEndBB);
}

// ((x & (x - 1)) == 0)
llvm::Value* is_log2(llvm::Value* x, IRBuilder<>& builder)
{
  auto xsub1 = builder.CreateNSWSub(x, builder.getInt32(1));
  auto mask = builder.CreateAnd(x, xsub1);
  auto result = builder.CreateICmpEQ(mask, builder.getInt32(0));
  return result;
}

//  float f = x;
//  return (*(int*)(&f) >> 23) - 127;
llvm::Value* fast_log2(llvm::Value* x, IRBuilder<>& builder)
{
  auto castdata = builder.CreateSIToFP(x, builder.getFloatTy());
  auto bitdata = builder.CreateBitCast(castdata, builder.getInt32Ty());
  auto ashrdata = builder.CreateAShr(bitdata, builder.getInt32(23));
  auto result = builder.CreateNSWAdd(ashrdata, builder.getInt32(-127));
  return result;
}

void WorkitemLoops::CreateVortexTMVar(
    llvm::Function* F, VortexTMData& tmdata)
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

  LoadInst* loadLx = builder.CreateLoad(nLx, "nl_x");
  LoadInst* loadLy = builder.CreateLoad(nLy, "nl_y");
  LoadInst* loadLz = builder.CreateLoad(nLz, "nl_z");
  auto loadLxy = builder.CreateBinOp(Instruction::Mul, loadLx, loadLy, "nl_xy");
  auto loadLxyz = builder.CreateBinOp(Instruction::Mul, loadLxy, loadLz, "nl_xyz");
  /*{{{*/
  /*{
    GlobalVariable* nGx = M->getGlobalVariable("_num_groups_x");
    if (nGx == NULL)
    nGx = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
    NULL, "_num_groups_x", NULL,
    GlobalValue::ThreadLocalMode::NotThreadLocal,
    0, true);

    GlobalVariable* nGy = M->getGlobalVariable("_num_groups_y");
    if (nGy == NULL)
    nGy = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
    NULL, "_num_groups_y", NULL,
    GlobalValue::ThreadLocalMode::NotThreadLocal,
    0, true);

    GlobalVariable* nGz = M->getGlobalVariable("_num_groups_z");
    if (nGz == NULL)
    nGz = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
    NULL, "_num_groups_z", NULL,
    GlobalValue::ThreadLocalMode::NotThreadLocal,
    0, true);
    LoadInst* loadGx = builder.CreateLoad(nGx, "ng_x");
    LoadInst* loadGy = builder.CreateLoad(nGy, "ng_y");
    auto loadGxy = builder.CreateBinOp(Instruction::Mul, loadGx, loadGy, "ng_xy");

    GlobalVariable* gIDx = M->getGlobalVariable("_group_id_x");
    GlobalVariable* gIDy = M->getGlobalVariable("_group_id_y");
    GlobalVariable* gIDz = M->getGlobalVariable("_group_id_z");

    LoadInst* loadGIDx = builder.CreateLoad(gIDx, "gid_x");
    LoadInst* loadGIDy = builder.CreateLoad(gIDy, "gid_y");
    LoadInst* loadGIDz = builder.CreateLoad(gIDz, "gid_z");
    auto tempIDy = builder.CreateBinOp(Instruction::Mul, loadGIDy, loadGx);
    auto tempIDz = builder.CreateBinOp(Instruction::Mul, loadGIDz, loadGxy);
    auto tempIDxy = builder.CreateBinOp(Instruction::Add, loadGIDx, tempIDy);
    auto gID = builder.CreateBinOp(Instruction::Add, tempIDxy, tempIDz, "gid");
    }*/
  /*}}}*/
  // Generate function def for getting VX function
  LLVMContext& context = M->getContext();
  FunctionType* nTTy = FunctionType::get(IntegerType::getInt32Ty(context), true);

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

  //auto remains = builder.CreateURem(loadLxyz, nHT, "remains");
  //auto loopWorks = builder.CreateSub(loadLxyz, remains, "loop_works");
  auto localIDHolder = builder.CreateAlloca(SizeT, 0, ".pocl.vortex_local_id");

  tmdata.LocalID = tlid;
  tmdata.TpC = TpC;
  tmdata.workload = loadLxyz;
  //tmdata.loopWorks = loopWorks;
  //tmdata.remains = remains;
  tmdata.num_local_x = loadLx;
  tmdata.num_local_xy = loadLxy;
  tmdata.localIDHolder = localIDHolder;

  return;
}

  std::pair<llvm::BasicBlock*, llvm::BasicBlock*>
WorkitemLoops::CreateVortexTMLoop(ParallelRegion& region,
    llvm::BasicBlock* entryBB, llvm::BasicBlock* exitBB,
    VortexTMData tmdata)
{
  auto lid = tmdata.LocalID;
  auto TpC = tmdata.TpC;
  auto workload = tmdata.workload;
  //auto loopWorks = tmdata.loopWorks;
  //auto remains = tmdata.remains;
  auto num_local_x = tmdata.num_local_x;
  auto num_local_xy = tmdata.num_local_xy;
  auto localIDHolder = tmdata.localIDHolder;

  llvm::BasicBlock* loopBodyEntryBB = entryBB;
  llvm::LLVMContext& ctx = loopBodyEntryBB->getContext();
  llvm::Function* F = loopBodyEntryBB->getParent();
  auto M = F->getParent();
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
  auto add = builder.CreateAdd(builder.CreateLoad(localIDHolder), TpC);
  builder.CreateStore(add, localIDHolder);
  builder.CreateBr(oldEntry);

  // Add instruction for Cond Block
  builder.SetInsertPoint(forCondBB);
  auto curid = builder.CreateLoad(localIDHolder);
  llvm::Value* cmpResult = builder.CreateICmpULT(curid, workload);
  Instruction* loopBranch = builder.CreateCondBr(
      cmpResult, loopBodyEntryBB, loopEndBB);

  // Replace usage of local id to vortex drived local id
  {
    builder.SetInsertPoint(&(loopBodyEntryBB->front()));
    auto curid = builder.CreateLoad(localIDHolder);

    // simple version
    auto local_id_z = builder.CreateUDiv(curid, num_local_xy, "new_lid_z");
    auto remains_xy = builder.CreateSub(curid,
        builder.CreateMul(local_id_z, num_local_xy));
    auto local_id_y = builder.CreateUDiv(remains_xy, num_local_x, "new_lid_y");
    auto local_id_x = builder.CreateSub(remains_xy,
        builder.CreateMul(local_id_y, num_local_x), "new_lid_x");

    auto M = F->getParent();
    std::vector<Instruction*> trashs;

    for (auto I = region.begin(); I != region.end(); ++I) {
      llvm::BasicBlock* bb = *I;
      std::cerr << "### VortexLoopGen : pred Name : " << bb->getName().str()
        << std::endl;

      if (!DT->dominates(loopBodyEntryBB, bb))
        continue;

      for (BasicBlock::iterator BI = bb->begin(); BI != bb->end(); ++BI) {
        Instruction* Instr = dyn_cast<Instruction>(BI);

        if (isa<llvm::LoadInst>(Instr)) {
          llvm::LoadInst* load = dyn_cast<llvm::LoadInst>(Instr);
          llvm::Value* pointer = load->getPointerOperand();
          StringRef pname = pointer->getName();

          std::cerr << "### VortexLoopGen : " << pname.str()
            << " : " << pname.equals("_local_id_x") << std::endl;

          if (pname.equals("_local_id_x")) {
            Instr->replaceAllUsesWith(local_id_x);
            trashs.push_back(Instr);
          } else if (pname.equals("_local_id_y")) {
            Instr->replaceAllUsesWith(local_id_y);
            trashs.push_back(Instr);
          } else if (pname.equals("_local_id_z")) {
            Instr->replaceAllUsesWith(local_id_z);
            trashs.push_back(Instr);
          }
        }
      }
    }
    for (auto BI : trashs) {
      BI->eraseFromParent();
    }
  }

  MDNode* Dummy
    = MDNode::getTemporary(ctx, ArrayRef<Metadata*>()).release();
#ifdef LLVM_OLDER_THAN_8_0
  MDNode* Root = MDNode::get(C, Dummy);
#else
  MDNode* AccessGroupMD = MDNode::getDistinct(ctx, {});
  MDNode* ParallelAccessMD = MDNode::get(
      ctx, { MDString::get(ctx, "llvm.loop.parallel_accesses"), AccessGroupMD });

  MDNode* Root = MDNode::get(ctx, { Dummy, ParallelAccessMD });
#endif

  // At this point we have
  //   !0 = metadata !{}            <- dummy
  //   !1 = metadata !{metadata !0} <- root
  // Replace the dummy operand with the root node itself and delete the dummy.
  Root->replaceOperandWith(0, Root);
  MDNode::deleteTemporary(Dummy);
  // We now have
  //   !1 = metadata !{metadata !1} <- self-referential root
  loopBranch->setMetadata("llvm.loop", Root);

#ifdef LLVM_OLDER_THAN_8_0
  region.AddParallelLoopMetadata(Root);
#else
  region.AddParallelLoopMetadata(AccessGroupMD);
#endif

  builder.SetInsertPoint(loopEndBB);
  builder.CreateBr(oldExit);

  return std::make_pair(forInitBB, loopEndBB);
}

  ParallelRegion*
WorkitemLoops::RegionOfBlock(llvm::BasicBlock* bb)
{
  for (ParallelRegion::ParallelRegionVector::iterator
      i
      = original_parallel_regions->begin(),
      e = original_parallel_regions->end();
      i != e; ++i) {
    ParallelRegion* region = (*i);
    if (region->HasBlock(bb))
      return region;
  }
  return NULL;
}

void WorkitemLoops::releaseParallelRegions()
{
  if (original_parallel_regions) {
    for (auto i = original_parallel_regions->begin(),
        e = original_parallel_regions->end();
        i != e; ++i) {
      ParallelRegion* p = *i;
      delete p;
    }

    delete original_parallel_regions;
    original_parallel_regions = nullptr;
  }
}

bool WorkitemLoops::ProcessFunction(Function& F)
{
  Kernel* K = cast<Kernel>(&F);

  llvm::Module* M = K->getParent();

  Initialize(K);
  unsigned workItemCount = WGLocalSizeX * WGLocalSizeY * WGLocalSizeZ;

  if (workItemCount == 1) {
    K->addLocalSizeInitCode(WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ);
    ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);
    return true;
  }

  releaseParallelRegions();

  original_parallel_regions = K->getParallelRegions(&LI->getLoopInfo());

#ifdef DUMP_CFGS
  //F.dump();
  //printIR(&F);
  dumpCFG(F, F.getName().str() + "_before_wiloops.dot",
      original_parallel_regions);
#endif

  IRBuilder<> builder(&*(F.getEntryBlock().getFirstInsertionPt()));
  localIdXFirstVar = builder.CreateAlloca(SizeT, 0, ".pocl.local_id_x_init");

  bool TMForVortex = false;
#ifdef BUILD_VORTEX
  TMForVortex = (VORTEX_MAPPING == 1);
#endif

  VortexTMData tmdata;
  if (TMForVortex) {
    CreateVortexTMVar(&F, tmdata);
  }
#if 0
  std::cerr << "### Original" << std::endl;
  F.viewCFGOnly();
#endif

#if 0
  for (ParallelRegion::ParallelRegionVector::iterator
      i = original_parallel_regions->begin(),
      e = original_parallel_regions->end();
      i != e; ++i)
  {
    ParallelRegion *region = (*i);
    region->InjectRegionPrintF();
    region->InjectVariablePrintouts();
  }
#endif

  /* Count how many parallel regions share each entry node to
     detect diverging regions that need to be peeled. */
  std::map<llvm::BasicBlock*, int> entryCounts;

  for (ParallelRegion::ParallelRegionVector::iterator
      i
      = original_parallel_regions->begin(),
      e = original_parallel_regions->end();
      i != e; ++i) {
    ParallelRegion* region = (*i);
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### Adding context save/restore for PR: ";
    region->dumpNames();
#endif
    FixMultiRegionVariables(region);
    entryCounts[region->entryBB()]++;
  }

#if 0
  std::cerr << "### After context code addition:" << std::endl;
  F.viewCFG();
#endif
  std::map<ParallelRegion*, bool> peeledRegion;
  for (ParallelRegion::ParallelRegionVector::iterator
      i
      = original_parallel_regions->begin(),
      e = original_parallel_regions->end();
      i != e; ++i) {
    llvm::ValueToValueMapTy reference_map;
    ParallelRegion* original = (*i);

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### handling region:" << std::endl;
    original->dumpNames();
    //F.viewCFGOnly();
#endif

    /* In case of conditional barriers, the first iteration
       has to be peeled so we know which branch to execute
       with the work item loop. In case there are more than one
       parallel region sharing an entry BB, it's a diverging
       region.

       Post dominance of entry by exit does not work in case the
       region is inside a loop and the exit block is in the path
       towards the loop exit (and the function exit).
       */
    bool peelFirst = entryCounts[original->entryBB()] > 1;

    peeledRegion[original] = peelFirst;

    std::pair<llvm::BasicBlock*, llvm::BasicBlock*> l;
    // the original predecessor nodes of which successor
    // should be fixed if not peeling
    BasicBlockVector preds;

    bool unrolled = false;
    if (peelFirst) {
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### conditional region, peeling the first iteration" << std::endl;
#endif
      ParallelRegion* replica = original->replicate(reference_map, ".peeled_wi");
      replica->chainAfter(original);
      replica->purge();

      l = std::make_pair(replica->entryBB(), replica->exitBB());
    } else {
      llvm::pred_iterator PI = llvm::pred_begin(original->entryBB()),
        E = llvm::pred_end(original->entryBB());

      for (; PI != E; ++PI) {
        llvm::BasicBlock* bb = *PI;
        if (DT->dominates(original->entryBB(), bb) && (RegionOfBlock(original->entryBB()) == RegionOfBlock(bb)))
          continue;
        preds.push_back(bb);
      }

      unsigned unrollCount;
      if (getenv("POCL_WILOOPS_MAX_UNROLL_COUNT") != NULL)
        unrollCount = atoi(getenv("POCL_WILOOPS_MAX_UNROLL_COUNT"));
      else
        unrollCount = 1;
      /* Find a two's exponent unroll count, if available. */
      while (unrollCount >= 1) {
        if (WGLocalSizeX % unrollCount == 0 && unrollCount <= WGLocalSizeX) {
          break;
        }
        unrollCount /= 2;
      }

      if (unrollCount > 1) {
        ParallelRegion* prev = original;
        llvm::BasicBlock* lastBB = AppendIncBlock(original->exitBB(), LocalIdXGlobal);
        original->AddBlockAfter(lastBB, original->exitBB());
        original->SetExitBB(lastBB);

        if (AddWIMetadata)
          original->AddIDMetadata(F.getContext(), 0);

        for (unsigned c = 1; c < unrollCount; ++c) {
          ParallelRegion* unrolled = original->replicate(reference_map, ".unrolled_wi");
          unrolled->chainAfter(prev);
          prev = unrolled;
          lastBB = unrolled->exitBB();
          if (AddWIMetadata)
            unrolled->AddIDMetadata(F.getContext(), c);
        }
        unrolled = true;
        l = std::make_pair(original->entryBB(), lastBB);
      } else {
        l = std::make_pair(original->entryBB(), original->exitBB());
      }
    }

    if (TMForVortex) {
      l = CreateVortexTMLoop(*original, l.first, l.second, tmdata);
    } else if (WGDynamicLocalSize) {
      GlobalVariable* gv;
      gv = M->getGlobalVariable("_local_size_x");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
            NULL, "_local_size_x", NULL,
            GlobalValue::ThreadLocalMode::NotThreadLocal,
            0, true);

      l = CreateLoopAround(*original, l.first, l.second, peelFirst,
          LocalIdXGlobal, WGLocalSizeX, !unrolled, gv);

      gv = M->getGlobalVariable("_local_size_y");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, false, GlobalValue::CommonLinkage,
            NULL, "_local_size_y");

      l = CreateLoopAround(*original, l.first, l.second,
          false, LocalIdYGlobal, WGLocalSizeY, !unrolled, gv);

      gv = M->getGlobalVariable("_local_size_z");
      if (gv == NULL)
        gv = new GlobalVariable(*M, SizeT, true, GlobalValue::CommonLinkage,
            NULL, "_local_size_z", NULL,
            GlobalValue::ThreadLocalMode::NotThreadLocal,
            0, true);

      l = CreateLoopAround(*original, l.first, l.second,
          false, LocalIdZGlobal, WGLocalSizeZ, !unrolled, gv);
      printIR(&F);
    } else {
      if (WGLocalSizeX > 1) {
        l = CreateLoopAround(*original, l.first, l.second, peelFirst,
            LocalIdXGlobal, WGLocalSizeX, !unrolled);
      }

      if (WGLocalSizeY > 1) {
        l = CreateLoopAround(*original, l.first, l.second, false,
            LocalIdYGlobal, WGLocalSizeY);
      }

      if (WGLocalSizeZ > 1) {
        l = CreateLoopAround(*original, l.first, l.second, false,
            LocalIdZGlobal, WGLocalSizeZ);
      }
    }

    /* Loop edges coming from another region mean B-loops which means
       we have to fix the loop edge to jump to the beginning of the wi-loop
       structure, not its body. This has to be done only for non-peeled
       blocks as the semantics is correct in the other case (the jump is
       to the beginning of the peeled iteration). */
    if (!peelFirst) {
      for (BasicBlockVector::iterator i = preds.begin();
          i != preds.end(); ++i) {
        llvm::BasicBlock* bb = *i;
        bb->getTerminator()->replaceUsesOfWith(original->entryBB(), l.first);
      }
    }
  }

  // for the peeled regions we need to add a prologue
  // that initializes the local ids and the first iteration
  // counter
  for (ParallelRegion::ParallelRegionVector::iterator
      i
      = original_parallel_regions->begin(),
      e = original_parallel_regions->end();
      i != e; ++i) {
    ParallelRegion* pr = (*i);

    if (!peeledRegion[pr])
      continue;
    pr->insertPrologue(0, 0, 0);
    builder.SetInsertPoint(&*(pr->entryBB()->getFirstInsertionPt()));
    builder.CreateStore(ConstantInt::get(SizeT, 1), localIdXFirstVar);
  }

  if (!WGDynamicLocalSize)
    K->addLocalSizeInitCode(WGLocalSizeX, WGLocalSizeY, WGLocalSizeZ);

  ParallelRegion::insertLocalIdInit(&F.getEntryBlock(), 0, 0, 0);

#if 0
  F.viewCFG();
#endif

  return true;
}

/*
 * Add context save/restore code to variables that are defined in
 * the given region and are used outside the region.
 *
 * Each such variable gets a slot in the stack frame. The variable
 * is restored from the stack whenever it's used.
 *
 */
void WorkitemLoops::FixMultiRegionVariables(ParallelRegion* region)
{
  InstructionIndex instructionsInRegion;
  InstructionVec instructionsToFix;

  /* Construct an index of the region's instructions so it's
     fast to figure out if the variable uses are all
     in the region. */
  for (BasicBlockVector::iterator i = region->begin();
      i != region->end(); ++i) {
    llvm::BasicBlock* bb = *i;
    for (llvm::BasicBlock::iterator instr = bb->begin();
        instr != bb->end(); ++instr) {
      llvm::Instruction* instruction = &*instr;
      instructionsInRegion.insert(instruction);
    }
  }

  /* Find all the instructions that define new values and
     check if they need to be context saved. */
  for (BasicBlockVector::iterator i = region->begin();
      i != region->end(); ++i) {
    llvm::BasicBlock* bb = *i;
    for (llvm::BasicBlock::iterator instr = bb->begin();
        instr != bb->end(); ++instr) {
      llvm::Instruction* instruction = &*instr;

      if (ShouldNotBeContextSaved(&*instr))
        continue;

      for (Instruction::use_iterator ui = instruction->use_begin(),
          ue = instruction->use_end();
          ui != ue; ++ui) {
        llvm::Instruction* user = dyn_cast<Instruction>(ui->getUser());

        if (user == NULL)
          continue;
        // If the instruction is used outside this region inside another
        // region (not in a regionless BB like the B-loop construct BBs),
        // need to context save it.
        // Allocas (private arrays) should be privatized always. Otherwise
        // we end up reading the same array, but replicating the GEP to that.
        if (isa<AllocaInst>(instruction) || (instructionsInRegion.find(user) == instructionsInRegion.end() && RegionOfBlock(user->getParent()) != NULL)) {
          instructionsToFix.push_back(instruction);
          break;
        }
      }
    }
  }

  /* Finally, fix the instructions. */
  for (InstructionVec::iterator i = instructionsToFix.begin();
      i != instructionsToFix.end(); ++i) {
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### adding context/save restore for" << std::endl;
    (*i)->dump();
#endif
    llvm::Instruction* instructionToFix = *i;
    AddContextSaveRestore(instructionToFix);
  }
}

  llvm::Value*
WorkitemLoops::GetLinearWiIndex(llvm::IRBuilder<>& builder, llvm::Module* M,
    ParallelRegion* region)
{
  GlobalVariable* LocalSizeXPtr = cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_x", SizeT));
  GlobalVariable* LocalSizeYPtr = cast<GlobalVariable>(M->getOrInsertGlobal("_local_size_y", SizeT));

  assert(LocalSizeXPtr != NULL && LocalSizeYPtr != NULL);

  LoadInst* LoadX = builder.CreateLoad(LocalSizeXPtr, "ls_x");
  LoadInst* LoadY = builder.CreateLoad(LocalSizeYPtr, "ls_y");

  /* Form linear index from xyz coordinates:
     local_size_x * local_size_y * local_id_z  (z dimension)
     + local_size_x * local_id_y                 (y dimension)
     + local_id_x                                (x dimension)
     */
  Value* LocalSizeXTimesY = builder.CreateBinOp(Instruction::Mul, LoadX, LoadY, "ls_xy");

  Value* ZPart = builder.CreateBinOp(Instruction::Mul, LocalSizeXTimesY,
      region->LocalIDZLoad(),
      "tmp");

  Value* YPart = builder.CreateBinOp(Instruction::Mul, LoadX, region->LocalIDYLoad(),
      "ls_x_y");

  Value* ZYSum = builder.CreateBinOp(Instruction::Add, ZPart, YPart,
      "zy_sum");

  return builder.CreateBinOp(Instruction::Add, ZYSum, region->LocalIDXLoad(),
      "linear_xyz_idx");
}

  llvm::Instruction*
WorkitemLoops::AddContextSave(llvm::Instruction* instruction, llvm::Instruction* alloca)
{

  if (isa<AllocaInst>(instruction)) {
    /* If the variable to be context saved is itself an alloca,
       we have created one big alloca that stores the data of all the
       work-items and return pointers to that array. Thus, we need
       no initialization code other than the context data alloca itself. */
    return NULL;
  }

  /* Save the produced variable to the array. */
  BasicBlock::iterator definition = (dyn_cast<Instruction>(instruction))->getIterator();
  ++definition;
  while (isa<PHINode>(definition))
    ++definition;

  IRBuilder<> builder(&*definition);
  std::vector<llvm::Value*> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion* region = RegionOfBlock(instruction->getParent());
  assert("Adding context save outside any region produces illegal code." && region != NULL);

  if (WGDynamicLocalSize) {
    Module* M = alloca->getParent()->getParent()->getParent();
    gepArgs.push_back(GetLinearWiIndex(builder, M, region));
  } else {
    gepArgs.push_back(ConstantInt::get(SizeT, 0));
    gepArgs.push_back(region->LocalIDZLoad());
    gepArgs.push_back(region->LocalIDYLoad());
    gepArgs.push_back(region->LocalIDXLoad());
  }

  return builder.CreateStore(instruction, builder.CreateGEP(alloca, gepArgs));
}

llvm::Instruction* WorkitemLoops::AddContextRestore(llvm::Value* val,
    llvm::Instruction* alloca,
    bool PoclWrapperStructAdded,
    llvm::Instruction* before,
    bool isAlloca)
{
  assert(val != NULL);
  assert(alloca != NULL);
  IRBuilder<> builder(alloca);
  if (before != NULL) {
    builder.SetInsertPoint(before);
  } else if (isa<Instruction>(val)) {
    builder.SetInsertPoint(dyn_cast<Instruction>(val));
    before = dyn_cast<Instruction>(val);
  } else {
    assert(false && "Unknown context restore location!");
  }

  std::vector<llvm::Value*> gepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion* region = RegionOfBlock(before->getParent());
  assert("Adding context save outside any region produces illegal code." && region != NULL);

  if (WGDynamicLocalSize) {
    Module* M = alloca->getParent()->getParent()->getParent();
    gepArgs.push_back(GetLinearWiIndex(builder, M, region));
  } else {
    gepArgs.push_back(ConstantInt::get(SizeT, 0));
    gepArgs.push_back(region->LocalIDZLoad());
    gepArgs.push_back(region->LocalIDYLoad());
    gepArgs.push_back(region->LocalIDXLoad());
  }

  if (PoclWrapperStructAdded)
    gepArgs.push_back(
        ConstantInt::get(Type::getInt32Ty(alloca->getContext()), 0));

  llvm::Instruction* gep = dyn_cast<Instruction>(builder.CreateGEP(alloca, gepArgs));
  if (isAlloca) {
    /* In case the context saved instruction was an alloca, we created a
       context array with pointed-to elements, and now want to return a
       pointer to the elements to emulate the original alloca. */
    return gep;
  }
  return builder.CreateLoad(gep);
}

/**
 * Returns the context array (alloca) for the given Value, creates it if not
 * found.
 */
  llvm::Instruction*
WorkitemLoops::GetContextArray(llvm::Instruction* instruction,
    bool& PoclWrapperStructAdded)
{
  PoclWrapperStructAdded = false;
  /*
   * Unnamed temp instructions need a generated name for the
   * context array. Create one using a running integer.
   */
  std::ostringstream var;
  var << ".";

  if (std::string(instruction->getName().str()) != "") {
    var << instruction->getName().str();
  } else if (tempInstructionIds.find(instruction) != tempInstructionIds.end()) {
    var << tempInstructionIds[instruction];
  } else {
    tempInstructionIds[instruction] = tempInstructionIndex++;
    var << tempInstructionIds[instruction];
  }

  var << ".pocl_context";
  std::string varName = var.str();

  if (contextArrays.find(varName) != contextArrays.end())
    return contextArrays[varName];

  BasicBlock& bb = instruction->getParent()->getParent()->getEntryBlock();
  IRBuilder<> builder(&*(bb.getFirstInsertionPt()));

  llvm::Type* elementType;
  if (isa<AllocaInst>(instruction)) {
    /* If the variable to be context saved was itself an alloca,
       create one big alloca that stores the data of all the
       work-items and directly return pointers to that array.
       This enables moving all the allocas to the entry node without
       breaking the parallel loop.
       Otherwise we would rely on a dynamic alloca to allocate
       unique stack space to all the work-items when its wiloop
       iteration is executed. */
    elementType = dyn_cast<AllocaInst>(instruction)->getType()->getElementType();
  } else {
    elementType = instruction->getType();
  }

  Module* M = instruction->getParent()->getParent()->getParent();
  const llvm::DataLayout& Layout = M->getDataLayout();

  /* 3D context array. In case the elementType itself is an array or struct,
   * we must take into account it could be alloca-ed with alignment and loads
   * or stores might use vectorized instructions expecting proper alignment.
   * Because of that, we cannot simply allocate x*y*z*(size), we must
   * enlarge the type to fit the alignment. */
  Type* AllocType = elementType;
  AllocaInst* InstCast = dyn_cast<AllocaInst>(instruction);
  if (InstCast) {
    unsigned Alignment = InstCast->getAlignment();

    uint64_t StoreSize = Layout.getTypeStoreSize(InstCast->getAllocatedType());

    if ((Alignment > 1) && (StoreSize & (Alignment - 1))) {
      uint64_t AlignedSize = (StoreSize & (~(Alignment - 1))) + Alignment;
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### unaligned type found: aligning " << StoreSize << " to "
        << AlignedSize << "\n";
#endif
      assert(AlignedSize > StoreSize);
      uint64_t RequiredExtraBytes = AlignedSize - StoreSize;

      if (isa<ArrayType>(elementType)) {

        ArrayType* StructPadding = ArrayType::get(
            Type::getInt8Ty(M->getContext()), RequiredExtraBytes);

        std::vector<Type*> PaddedStructElements;
        PaddedStructElements.push_back(elementType);
        PaddedStructElements.push_back(StructPadding);
        const ArrayRef<Type*> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(M->getContext(), NewStructElements, true);
        PoclWrapperStructAdded = true;
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);

      } else if (isa<StructType>(elementType)) {
        StructType* OldStruct = dyn_cast<StructType>(elementType);

        ArrayType* StructPadding = ArrayType::get(Type::getInt8Ty(M->getContext()), RequiredExtraBytes);
        std::vector<Type*> PaddedStructElements;
        for (unsigned j = 0; j < OldStruct->getNumElements(); j++)
          PaddedStructElements.push_back(OldStruct->getElementType(j));
        PaddedStructElements.push_back(StructPadding);
        const ArrayRef<Type*> NewStructElements(PaddedStructElements);
        AllocType = StructType::get(OldStruct->getContext(), NewStructElements,
            OldStruct->isPacked());
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);
      }
    }
  }

  llvm::AllocaInst* Alloca = nullptr;
  if (WGDynamicLocalSize) {
    char GlobalName[32];
    GlobalVariable* LocalSize;
    LoadInst* LocalSizeLoad[3];
    for (int i = 0; i < 3; ++i) {
      snprintf(GlobalName, 32, "_local_size_%c", 'x' + i);
      LocalSize = cast<GlobalVariable>(M->getOrInsertGlobal(GlobalName, SizeT));
      LocalSizeLoad[i] = builder.CreateLoad(LocalSize);
    }

    Value* LocalXTimesY = builder.CreateBinOp(Instruction::Mul, LocalSizeLoad[0],
        LocalSizeLoad[1], "tmp");
    Value* NumberOfWorkItems = builder.CreateBinOp(Instruction::Mul, LocalXTimesY,
        LocalSizeLoad[2], "num_wi");

    Alloca = builder.CreateAlloca(AllocType, NumberOfWorkItems, varName);
  } else {
    llvm::Type* contextArrayType = ArrayType::get(
        ArrayType::get(ArrayType::get(AllocType, WGLocalSizeX), WGLocalSizeY),
        WGLocalSizeZ);

    /* Allocate the context data array for the variable. */
    Alloca = builder.CreateAlloca(contextArrayType, nullptr, varName);
  }

  /* Align the context arrays to stack to enable wide vectors
     accesses to them. Also, LLVM 3.3 seems to produce illegal
     code at least with Core i5 when aligned only at the element
     size. */
  Alloca->setAlignment(
#ifndef LLVM_OLDER_THAN_10_0
      llvm::MaybeAlign(
#endif
        CONTEXT_ARRAY_ALIGN
#ifndef LLVM_OLDER_THAN_10_0
        )
#endif
      );

  contextArrays[varName] = Alloca;
  return Alloca;
}

/**
 * Adds context save/restore code for the value produced by the
 * given instruction.
 *
 * TODO: add only one restore per variable per region.
 * TODO: add only one load of the id variables per region.
 * Could be done by having a context restore BB in the beginning of the
 * region and a context save BB at the end.
 * TODO: ignore work group variables completely (the iteration variables)
 * The LLVM should optimize these away but it would improve
 * the readability of the output during debugging.
 * TODO: rematerialize some values such as extended values of global
 * variables (especially global id which is computed from local id) or kernel
 * argument values instead of allocating stack space for them
 */
void WorkitemLoops::AddContextSaveRestore(llvm::Instruction* instruction)
{

  /* Allocate the context data array for the variable. */
  bool PoclWrapperStructAdded = false;
  llvm::Instruction* alloca = GetContextArray(instruction, PoclWrapperStructAdded);
  llvm::Instruction* theStore = AddContextSave(instruction, alloca);

  InstructionVec uses;
  /* Restore the produced variable before each use to ensure the correct context
     copy is used.

     We could add the restore only to other regions outside the
     variable defining region and use the original variable in the defining
     region due to the SSA virtual registers being unique. However,
     alloca variables can be redefined also in the same region, thus we
     need to ensure the correct alloca context position is written, not
     the original unreplicated one. These variables can be generated by
     volatile variables, private arrays, and due to the PHIs to allocas
     pass.
     */

  /* Find out the uses to fix first as fixing them invalidates
     the iterator. */
  for (Instruction::use_iterator ui = instruction->use_begin(),
      ue = instruction->use_end();
      ui != ue; ++ui) {
    llvm::Instruction* user = cast<Instruction>(ui->getUser());
    if (user == NULL)
      continue;
    if (user == theStore)
      continue;
    uses.push_back(user);
  }

  for (InstructionVec::iterator i = uses.begin(); i != uses.end(); ++i) {
    Instruction* user = *i;
    Instruction* contextRestoreLocation = user;
    /* If the user is in a block that doesn't belong to a region,
       the variable itself must be a "work group variable", that is,
       not dependent on the work item. Most likely an iteration
       variable of a for loop with a barrier. */
    if (RegionOfBlock(user->getParent()) == NULL)
      continue;

    PHINode* phi = dyn_cast<PHINode>(user);
    if (phi != NULL) {
      /* In case of PHI nodes, we cannot just insert the context
         restore code before it in the same basic block because it is
         assumed there are no non-phi Instructions before PHIs which
         the context restore code constitutes to. Add the context
         restore to the incomingBB instead.

         There can be values in the PHINode that are incoming
         from another region even though the decision BB is within the region.
         For those values we need to add the context restore code in the
         incoming BB (which is known to be inside the region due to the
         assumption of not having to touch PHI nodes in PRentry BBs).
         */

      /* PHINodes at region entries are broken down earlier. */
      assert("Cannot add context restore for a PHI node at the region entry!" && RegionOfBlock(phi->getParent())->entryBB() != phi->getParent());
#ifdef DEBUG_WORK_ITEM_LOOPS
      std::cerr << "### adding context restore code before PHI" << std::endl;
      user->dump();
      std::cerr << "### in BB:" << std::endl;
      user->getParent()->dump();
#endif
      BasicBlock* incomingBB = NULL;
      for (unsigned incoming = 0; incoming < phi->getNumIncomingValues();
          ++incoming) {
        Value* val = phi->getIncomingValue(incoming);
        BasicBlock* bb = phi->getIncomingBlock(incoming);
        if (val == instruction)
          incomingBB = bb;
      }
      assert(incomingBB != NULL);
      contextRestoreLocation = incomingBB->getTerminator();
    }
    llvm::Value* loadedValue = AddContextRestore(
        user, alloca, PoclWrapperStructAdded, contextRestoreLocation,
        isa<AllocaInst>(instruction));
    user->replaceUsesOfWith(instruction, loadedValue);

#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### done, the user was converted to:" << std::endl;
    user->dump();
#endif
  }
}

bool WorkitemLoops::ShouldNotBeContextSaved(llvm::Instruction* instr)
{
  /*
     _local_id loads should not be replicated as it leads to
     problems in conditional branch case where the header node
     of the region is shared across the branches and thus the
     header node's ID loads might get context saved which leads
     to egg-chicken problems.
     */
  if (isa<BranchInst>(instr))
    return true;

  llvm::LoadInst* load = dyn_cast<llvm::LoadInst>(instr);
  if (load != NULL && (load->getPointerOperand() == LocalIdZGlobal || load->getPointerOperand() == LocalIdYGlobal || load->getPointerOperand() == LocalIdXGlobal))
    return true;

  VariableUniformityAnalysis& VUA = getAnalysis<VariableUniformityAnalysis>();

  /* In case of uniform variables (same for all work-items),
     there is no point to create a context array slot for them,
     but just use the original value everywhere.

     Allocas are problematic: they include the de-phi induction
     variables of the b-loops. In those case each work item
     has a separate loop iteration variable in the LLVM IR but
     which is really a parallel region loop invariant. But
     because we cannot separate such loop invariant variables
     at this point sensibly, let's just replicate the iteration
     variable to each work item and hope the latter optimizations
     reduce them back to a single induction variable outside the
     parallel loop.
     */
  if (!VUA.shouldBePrivatized(instr->getParent()->getParent(), instr)) {
#ifdef DEBUG_WORK_ITEM_LOOPS
    std::cerr << "### based on VUA, not context saving:";
    instr->dump();
#endif
    return true;
  }

  return false;
}

  llvm::BasicBlock*
WorkitemLoops::AppendIncBlock(llvm::BasicBlock* after, llvm::Value* localIdVar)
{
  llvm::LLVMContext& C = after->getContext();

  llvm::BasicBlock* oldExit = after->getTerminator()->getSuccessor(0);
  assert(oldExit != NULL);

  llvm::BasicBlock* forIncBB = BasicBlock::Create(C, "pregion_for_inc", after->getParent());

  after->getTerminator()->replaceUsesOfWith(oldExit, forIncBB);

  IRBuilder<> builder(oldExit);

  builder.SetInsertPoint(forIncBB);
  /* Create the iteration variable increment */
  builder.CreateStore(builder.CreateAdd(
        builder.CreateLoad(localIdVar),
        ConstantInt::get(SizeT, 1)),
      localIdVar);

  builder.CreateBr(oldExit);

  return forIncBB;
}
