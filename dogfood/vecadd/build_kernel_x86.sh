#!/bin/bash

rm -rf ~/.cache/pocl
rm libvecadd_x86.a vecadd.pocl

POCL_DEVICES=basic POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=$PWD/../../../lvm_project/drops_x86/lib:$PWD/../../drops_x86_cc/lib $PWD/../../drops_x86_cc/bin/poclcc -l -o libvecadd_x86.a ./kernel.cl