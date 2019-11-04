#!/bin/bash

POCL_DEVICES=basic POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=/home/blaise/dev/pocl/drops_riscv/lib /home/blaise/dev/riscv-gnu-toolchain/drops/bin/qemu-riscv32 ./vecadd