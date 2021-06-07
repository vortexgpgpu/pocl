#!/bin/bash

KERNEL=$1

compile_x86()
{
    LLVM_PREFIX=/home/blaise/dev/llvm-riscv/releaseX86
    POCL_CC_PATH=/home/blaise/dev/pocl/release/rv86cc
    POCL_RT_PATH=/home/blaise/dev/pocl/release/rv86rt

    @rm -rf ~/.cache/pocl/

    LLVM_PREFIX=${LLVM_PREFIX} POCL_DEBUG=all LD_LIBRARY_PATH=${LLVM_PREFIX}/lib:${POCL_CC_PATH}/lib ${POCL_CC_PATH}/bin/poclcc -o ${KERNEL}.pocl ${KERNEL}.cl
}

compile_riscv_lx()
{
    RISCV_TOOLCHAIN_PATH=/opt/riscv-gnu-toolchain
    LLVM_PREFIX=/home/blaise/dev/llvm-riscv/releaseRV32
    POCL_CC_PATH=/home/blaise/dev/pocl/release/riscv_lx_cc
    POCL_RT_PATH=/home/blaise/dev/pocl/release/riscv_lx_rt

    @rm -rf ~/.cache/pocl/

    LLVM_PREFIX=${LLVM_PREFIX} POCL_DEBUG=all LD_LIBRARY_PATH=${LLVM_PREFIX}/lib:${POCL_CC_PATH}/lib ${POCL_CC_PATH}/bin/poclcc -o ${KERNEL}.pocl ${KERNEL}.cl
}

compile_riscv_lx_spv()
{
    RISCV_TOOLCHAIN_PATH=/home/blaise/dev/riscv-gnu-toolchain/release_linux64
    LLVM_PREFIX=/home/blaise/dev/llvm-riscv/releaseRV64
    POCL_CC_PATH=/home/blaise/dev/pocl/drops_rvlx64_cc
    POCL_RT_PATH=/home/blaise/dev/pocl/drops_rvlx64_rt

    @rm -rf ~/.cache/pocl/

    LLVM_PREFIX=${LLVM_PREFIX} POCL_DEBUG=all LD_LIBRARY_PATH=${LLVM_PREFIX}/lib:${POCL_CC_PATH}/lib ${POCL_CC_PATH}/bin/poclcc -o ${KERNEL}.pocl -TYPE SPIRV ${KERNEL}.spv
}

compile_riscv_lx_spv_exe()
{
    RISCV_TOOLCHAIN_PATH=/home/blaise/dev/riscv-gnu-toolchain/release_linux64
    LLVM_PREFIX=/home/blaise/dev/llvm-riscv/releaseRV64
    POCL_CC_PATH=/home/blaise/dev/pocl/drops_rvlx64_cc
    POCL_RT_PATH=/home/blaise/dev/pocl/drops_rvlx64_rt

    @rm -rf ~/.cache/pocl/

    LLVM_PREFIX=${LLVM_PREFIX} POCL_DEBUG=all LD_LIBRARY_PATH=${LLVM_PREFIX}/lib:${POCL_CC_PATH}/lib ${POCL_CC_PATH}/bin/poclcc -o ${KERNEL}.pocl -TYPE SPIRV ${KERNEL}.spv
}

run_x86() 
{
    POCL_RT_PATH=/home/blaise/dev/pocl/release/rv86rt
    LD_LIBRARY_PATH=$POCL_RT_PATH/lib:$LD_LIBRARY_PATH ./vecadd -k kernel.pocl
}

run_x86() 
{
    POCL_RT_PATH=opencl
    LD_LIBRARY_PATH=$POCL_RT_PATH/lib:/usr/lib64:$LD_LIBRARY_PATH ./vecadd -k kernel.pocl
}

compile_riscv_lx_spv