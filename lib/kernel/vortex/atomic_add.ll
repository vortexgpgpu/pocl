define i32 @_Z14_cl_atomic_addPU8CLglobalVii(ptr noundef %0, i32 noundef %1){
  %3 = atomicrmw volatile add ptr %0, i32 %1 acq_rel, align 4
  ret i32 %3
}

define float @_Z14_cl_atomic_addPU8CLglobalVff(ptr noundef %0, float noundef %1) {
  %3 = atomicrmw fadd ptr %0, float %1 acq_rel, align 4
  ret float %3
}
 
define float @_Z14_cl_atomic_addPU8CLglobalVif(ptr noundef %0, float noundef %1) {
  %3 = atomicrmw fadd ptr %0, float %1 acq_rel, align 4
  ret float %3
} 
