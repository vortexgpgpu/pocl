#!/bin/bash

$PWD/../../../../llvm/drops_x86/bin/clang -g -L$PWD/../../../drops_x86_rt/lib/static -L. ../vecadd.c -lOpenCL -Wl,--whole-archive -lvecadd -Wl,--no-whole-archive -o vecadd