#!/bin/bash

POCL_DEVICES=basic POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=~/dev/riscv-gnu-toolchain/drops/lib:/home/blaise/dev/pocl/drops_riscv_cc/lib /home/blaise/dev/pocl/drops_riscv_cc/bin/poclcc -o vecadd.pocl vecadd.cl

#LD_LIBRARY_PATH=~/dev/llvm_riscv/drops_x86/lib:~/dev/pocl/drops_x86_cc/lib ./bin/poclcc -l