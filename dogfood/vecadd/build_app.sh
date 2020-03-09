#!/bin/bash

#$PWD/../../../riscv-gnu-toolchain/drops/bin/riscv32-unknown-elf-gcc -g -O0 -I$PWD/../../drops_riscv_rt/include -L$PWD/../../drops_riscv_rt/lib/static -L. vecadd.c -lOpenCL -Wl,--whole-archive -lvecadd -Wl,--no-whole-archive -o vecadd
$PWD/../../../riscv-gnu-toolchain/drops/bin/clang -g -O0 -I$PWD/../../drops_riscv_rt/include -L$PWD/../../drops_riscv_rt/lib/static -L. vecadd.c -lOpenCL -Wl,--whole-archive -lvecadd -Wl,--no-whole-archive -o vecadd