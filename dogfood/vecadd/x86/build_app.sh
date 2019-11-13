#!/bin/bash

~/dev/llvm_riscv/drops_x86/bin/clang -g -L/home/blaise/dev/pocl/drops_x86_rt/lib/static -L. ../vecadd.c -lOpenCL -Wl,--whole-archive -lvecadd -Wl,--no-whole-archive -o vecadd