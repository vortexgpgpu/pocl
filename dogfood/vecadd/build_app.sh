#!/bin/bash

/home/blaise/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-gcc -g -O0 -I/home/blaise/dev/pocl/drops_riscv_rt/include -L/home/blaise/dev/pocl/drops_riscv_rt/lib/static -L. vecadd.c -lOpenCL -Wl,--whole-archive -lvecadd -Wl,--no-whole-archive -o vecadd