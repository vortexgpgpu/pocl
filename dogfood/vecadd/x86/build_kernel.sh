#!/bin/bash

rm -rf ~/.cache/pocl
rm libvecadd.a vecadd.pocl
POCL_DEVICES=basic POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=~/dev/llvm_riscv/drops_x86/lib:~/dev/pocl/drops_x86_cc/lib /home/blaise/dev/pocl/drops_x86_cc/bin/poclcc -l -o vecadd.pocl ../vecadd.cl