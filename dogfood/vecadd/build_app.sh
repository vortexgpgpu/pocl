#!/bin/bash

#g++ -g -Wall -I/home/blaise/dev/pocl/drops_riscv/include -L/home/blaise/dev/pocl_riscv/drops/lib -lstdc++ -lm -lOpenCL -gdwarf-2 -fno-rtti vecadd.c -ovecadd

/home/blaise/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-gcc -I/home/blaise/dev/pocl/drops_riscv_rt/include -L/home/blaise/dev/pocl/drops_riscv_rt/lib/static -L. vecadd.c -lOpenCL -Wl,--whole-archive -lvecadd -Wl,--no-whole-archive -o vecadd