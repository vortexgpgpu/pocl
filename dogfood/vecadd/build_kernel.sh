#!/bin/bash

rm -rf ~/.cache/pocl
rm libvecadd.a vecadd.pocl

POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=$RISCV_GNU_TOOLS_PATH/lib:$PWD/../../drops_riscv_cc/lib $PWD/../../drops_riscv_cc/bin/poclcc -o libvecadd.a kernel.cl