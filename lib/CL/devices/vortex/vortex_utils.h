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

#ifdef __cplusplus
}
#endif

#endif