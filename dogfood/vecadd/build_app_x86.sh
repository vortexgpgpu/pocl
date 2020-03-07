#!/bin/bash

$PWD/../../../llvm_project/drops_x86/bin/clang -g -L$PWD/../../drops_x86_rt/lib/static -L. ./vecadd.c -lOpenCL -Wl,--whole-archive -lvecadd_x86 -Wl,--no-whole-archive -o vecadd_x86