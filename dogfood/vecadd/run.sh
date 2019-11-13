#!/bin/bash

POCL_DEBUG=all /home/blaise/dev/riscv-gnu-toolchain/drops/bin/qemu-riscv32 -d in_asm -D debug.log ./vecadd