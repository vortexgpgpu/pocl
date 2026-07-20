#ifndef VORTEX_UTILS_H
#define VORTEX_UTILS_H

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

void remove_extension(char* filename);

/* module_slot: index of this program within its device context. Slot 0 links
 * at the default STARTUP_ADDR; slot N>0 is shifted so multiple co-resident
 * programs (e.g. hybridsort's bucketsort + mergesort) get non-overlapping
 * device code regions. */
int compile_vortex_program(char* sz_program_vxbin, void* llvm_module,
                           unsigned module_slot);

/* Non-zero if the program needs the RISC-V 'A' extension (LLVM atomics, or the
 * kernel library's inline amo/lr/sc asm, which -march cannot reject). The device
 * capability check lives in pocl_vortex_post_build_program. */
int vortex_module_uses_atomics(void* llvm_module);

#ifdef __cplusplus
}
#endif

#endif