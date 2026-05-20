#ifndef VORTEX_UTILS_H
#define VORTEX_UTILS_H

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

void remove_extension(char* filename);

int compile_vortex_program(char** kernel_names, int* num_kernels, char* sz_program_vxbin, void* llvm_module);

#ifdef __cplusplus
}
#endif

#endif