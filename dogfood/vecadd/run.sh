#!/bin/bash

POCL_DEBUG=all $PWD/../../../riscv-gnu-toolchain/drops/bin/qemu-riscv32 -d in_asm -D debug.log ./vecadd