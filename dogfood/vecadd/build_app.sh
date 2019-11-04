#!/bin/bash

#g++ -g -Wall -I/home/blaise/dev/pocl/drops_riscv/include -L/home/blaise/dev/pocl_riscv/drops/lib -lstdc++ -lm -lOpenCL -gdwarf-2 -fno-rtti vecadd.c -ovecadd
/home/blaise/dev/riscv-gnu-toolchain/drops/bin/riscv32-unknown-linux-gnu-g++ -g -Wall -I/home/blaise/dev/pocl/drops_riscv/include -L/home/blaise/dev/pocl/drops_riscv/lib -lm -lOpenCL vecadd.c -o vecadd