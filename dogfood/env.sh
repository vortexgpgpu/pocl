#!/bin/bash

export PATH=$PWD/../drops_riscv_cc/bin:$PWD/../../riscv-gnu-toolchain/drops/bin:$PATH
export LD_LIBRARY_PATH=$PWD/../drops_riscv_rt/lib/static:$PWD/../../riscv-gnu-toolchain/drops/lib:$LD_LIBRARY_PATH