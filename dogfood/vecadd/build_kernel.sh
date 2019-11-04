#!/bin/bash

POCL_DEVICES=basic POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=~/dev/riscv-gnu-toolchain/drops/lib:~/dev/pocl/drops_poclcc/lib  /home/blaise/dev/pocl/drops_poclcc/bin/poclcc -o vecadd.pocl vecadd.cl