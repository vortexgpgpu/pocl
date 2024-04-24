; ModuleID = 'atomics.cl.bc'
source_filename = "/nethome/sjeong306/pocl-4/lib/kernel/atomics.cl"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32"

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_addPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_addPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_subPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_subPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_incPU8CLglobalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_incPU8CLglobalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_decPU8CLglobalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_decPU8CLglobalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_andPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_andPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atomic_orPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z11_cl_atom_orPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_xorPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_xorPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z15_cl_atomic_xchgPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atom_xchgPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z18_cl_atomic_cmpxchgPU8CLglobalViii(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z16_cl_atom_cmpxchgPU8CLglobalViii(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_minPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile min ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_maxPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile max ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_minPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile min ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_maxPU8CLglobalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile max ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_addPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_addPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_subPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_subPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_incPU8CLglobalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_incPU8CLglobalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_decPU8CLglobalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_decPU8CLglobalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_andPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_andPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atomic_orPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z11_cl_atom_orPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_xorPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_xorPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z15_cl_atomic_xchgPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atom_xchgPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z18_cl_atomic_cmpxchgPU8CLglobalVjjj(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z16_cl_atom_cmpxchgPU8CLglobalVjjj(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_minPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umin ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_maxPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umax ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_minPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umin ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_maxPU8CLglobalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umax ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z15_cl_atomic_xchgPU8CLglobalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = bitcast float %1 to i32
  %4 = atomicrmw volatile xchg ptr %0, i32 %3 acq_rel, align 4
  %5 = bitcast i32 %4 to float
  ret float %5
}

; Function Attrs: nofree norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z14_cl_atomic_addPU8CLglobalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #1 {
  %3 = load volatile float, ptr %0, align 4, !tbaa !7
  %4 = bitcast float %3 to i32
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i32 [ %4, %2 ], [ %11, %5 ]
  %7 = bitcast i32 %6 to float
  %8 = fadd float %7, %1
  %9 = bitcast float %8 to i32
  %10 = cmpxchg volatile ptr %0, i32 %6, i32 %9 acq_rel acquire, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  %12 = extractvalue { i32, i1 } %10, 1
  br i1 %12, label %13, label %5

13:                                               ; preds = %5
  %14 = extractvalue { i32, i1 } %10, 0
  %15 = bitcast i32 %14 to float
  ret float %15
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z13_cl_atom_xchgPU8CLglobalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = bitcast float %1 to i32
  %4 = atomicrmw volatile xchg ptr %0, i32 %3 acq_rel, align 4
  %5 = bitcast i32 %4 to float
  ret float %5
}

; Function Attrs: nofree norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z12_cl_atom_addPU8CLglobalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #1 {
  %3 = load volatile float, ptr %0, align 4, !tbaa !7
  %4 = bitcast float %3 to i32
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i32 [ %4, %2 ], [ %11, %5 ]
  %7 = bitcast i32 %6 to float
  %8 = fadd float %7, %1
  %9 = bitcast float %8 to i32
  %10 = cmpxchg volatile ptr %0, i32 %6, i32 %9 acq_rel acquire, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  %12 = extractvalue { i32, i1 } %10, 1
  br i1 %12, label %13, label %5

13:                                               ; preds = %5
  %14 = extractvalue { i32, i1 } %10, 0
  %15 = bitcast i32 %14 to float
  ret float %15
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_addPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_addPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_subPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_subPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_incPU7CLlocalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_incPU7CLlocalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_decPU7CLlocalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_decPU7CLlocalVi(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_andPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_andPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atomic_orPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z11_cl_atom_orPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_xorPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_xorPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z15_cl_atomic_xchgPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atom_xchgPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z18_cl_atomic_cmpxchgPU7CLlocalViii(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z16_cl_atom_cmpxchgPU7CLlocalViii(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_minPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile min ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_maxPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile max ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_minPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile min ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_maxPU7CLlocalVii(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile max ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_addPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_addPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_subPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_subPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile sub ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_incPU7CLlocalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_incPU7CLlocalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile add ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_decPU7CLlocalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_decPU7CLlocalVj(ptr noundef %0) local_unnamed_addr #0 {
  %2 = atomicrmw volatile sub ptr %0, i32 1 acq_rel, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_andPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_andPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile and ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atomic_orPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z11_cl_atom_orPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile or ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_xorPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_xorPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xor ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z15_cl_atomic_xchgPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z13_cl_atom_xchgPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile xchg ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z18_cl_atomic_cmpxchgPU7CLlocalVjjj(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z16_cl_atom_cmpxchgPU7CLlocalVjjj(ptr noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = cmpxchg volatile ptr %0, i32 %1, i32 %2 acq_rel acquire, align 4
  %5 = extractvalue { i32, i1 } %4, 0
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_minPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umin ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z14_cl_atomic_maxPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umax ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_minPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umin ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define i32 @_Z12_cl_atom_maxPU7CLlocalVjj(ptr noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = atomicrmw volatile umax ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z15_cl_atomic_xchgPU7CLlocalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = bitcast float %1 to i32
  %4 = atomicrmw volatile xchg ptr %0, i32 %3 acq_rel, align 4
  %5 = bitcast i32 %4 to float
  ret float %5
}

; Function Attrs: nofree norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z14_cl_atomic_addPU7CLlocalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #1 {
  %3 = load volatile float, ptr %0, align 4, !tbaa !7
  %4 = bitcast float %3 to i32
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i32 [ %4, %2 ], [ %11, %5 ]
  %7 = bitcast i32 %6 to float
  %8 = fadd float %7, %1
  %9 = bitcast float %8 to i32
  %10 = cmpxchg volatile ptr %0, i32 %6, i32 %9 acq_rel acquire, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  %12 = extractvalue { i32, i1 } %10, 1
  br i1 %12, label %13, label %5

13:                                               ; preds = %5
  %14 = extractvalue { i32, i1 } %10, 0
  %15 = bitcast i32 %14 to float
  ret float %15
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z13_cl_atom_xchgPU7CLlocalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = bitcast float %1 to i32
  %4 = atomicrmw volatile xchg ptr %0, i32 %3 acq_rel, align 4
  %5 = bitcast i32 %4 to float
  ret float %5
}

; Function Attrs: nofree norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define float @_Z12_cl_atom_addPU7CLlocalVff(ptr noundef %0, float noundef %1) local_unnamed_addr #1 {
  %3 = load volatile float, ptr %0, align 4, !tbaa !7
  %4 = bitcast float %3 to i32
  br label %5

5:                                                ; preds = %5, %2
  %6 = phi i32 [ %4, %2 ], [ %11, %5 ]
  %7 = bitcast i32 %6 to float
  %8 = fadd float %7, %1
  %9 = bitcast float %8 to i32
  %10 = cmpxchg volatile ptr %0, i32 %6, i32 %9 acq_rel acquire, align 4
  %11 = extractvalue { i32, i1 } %10, 0
  %12 = extractvalue { i32, i1 } %10, 1
  br i1 %12, label %13, label %5

13:                                               ; preds = %5
  %14 = extractvalue { i32, i1 } %10, 0
  %15 = bitcast i32 %14 to float
  ret float %15
}

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "frame-pointer"="all" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+m,+relax,-d,-e,-experimental-zawrs,-experimental-zca,-experimental-zcd,-experimental-zcf,-experimental-zihintntl,-experimental-ztso,-experimental-zvfh,-f,-h,-save-restore,-svinval,-svnapot,-svpbmt,-v,-xtheadvdot,-xventanacondops,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zdinx,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zicbom,-zicbop,-zicboz,-zihintpause,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #1 = { nofree norecurse nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "frame-pointer"="all" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+c,+m,+relax,-d,-e,-experimental-zawrs,-experimental-zca,-experimental-zcd,-experimental-zcf,-experimental-zihintntl,-experimental-ztso,-experimental-zvfh,-f,-h,-save-restore,-svinval,-svnapot,-svpbmt,-v,-xtheadvdot,-xventanacondops,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zdinx,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zicbom,-zicbop,-zicboz,-zihintpause,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-zmmul,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!opencl.ocl.version = !{!5}
!llvm.ident = !{!6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"target-abi", !"ilp32"}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{i32 1, !"SmallDataLimit", i32 0}
!5 = !{i32 1, i32 2}
!6 = !{!"clang version 16.0.6 (git@github.com:vortexgpgpu/llvm.git 58811bfa61a503fd4a5f0dc7b57802fae51c3f5d)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"float", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
