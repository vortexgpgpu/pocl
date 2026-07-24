#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2018 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

include("bitcode_rules")

set(LLVM_TARGET_64 riscv64)
set(TARGET_MARCH_64 rv64imafd)

set(LLVM_TARGET_32 riscv32)
set(TARGET_MARCH_32 rv32imaf)

set(KERNEL_SOURCES ${SOURCES_GENERIC})

foreach(FILE printf.c printf_base.c
             get_work_dim.c get_num_groups.c get_local_size.c get_global_offset.c get_global_size.c
             get_group_id.c get_local_id.c get_global_id.c
             get_linear_id.cl
             barrier.ll
             atomics.cl)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE workitems.c printf.c barrier.c atomics.c)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
  list(APPEND KERNEL_SOURCES "vortex/${FILE}")
endforeach()

# libm entry points clang lowers to calls (no Vortex instruction backs them),
# so they must resolve inside the kernel bitcode library.
list(APPEND KERNEL_SOURCES "vortex/vxlibm.c")

# wait_group_events must be a real work-group barrier here: a Vortex work-group
# spans multiple hardware warps, so the generic empty stub (sound only for the
# work-item-loop model) lets non-copying warps read the data early.
list(REMOVE_ITEM KERNEL_SOURCES "wait_group_events.cl")
list(APPEND KERNEL_SOURCES "vortex/wait_group_events.cl")

# OpenCL image builtins (software). read_image/write_image use Vortex-specific
# scalar implementations (vortex/*.cl): the generic pocl versions rely on OpenCL
# vector arithmetic, which the no-vector Vortex backend (+xvortex) cannot select.
# The get_image_* queries are scalar and reuse the generic sources. Sampling runs
# in software over the dev_image_t descriptor; no fixed-function TEX hardware is
# required. See the pocl_image_support proposal.
foreach(FILE vortex/read_image.cl vortex/write_image.cl
             get_image_width.cl get_image_height.cl get_image_depth.cl
             get_image_dim.cl get_image_array_size.cl
             get_image_channel_data_type.cl get_image_channel_order.cl)
  list(APPEND KERNEL_SOURCES "${FILE}")
endforeach()

set(KERNEL_CL_FLAGS "-Xclang" "-cl-std=CL${VORTEX_DEVICE_CL_STD}" "-D__OPENCL_C_VERSION__=${VORTEX_DEVICE_CL_VERSION}" "-Xclang" "-cl-ext=all" ${KERNEL_CL_FLAGS})

set(LLC_FLAGS "")

set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${VORTEX_DEVICE_CL_VERSION}")
separate_arguments(VORTEX_DEVICE_EXTENSIONS)
foreach(EXT ${VORTEX_DEVICE_EXTENSIONS})
  set(DEVICE_CL_FLAGS "${DEVICE_CL_FLAGS} -D${EXT}")
endforeach()
separate_arguments(DEVICE_CL_FLAGS)

# Device-side Vortex SDK is consumed through per-XLEN install trees.
# Vortex builds are per-XLEN (one build dir per --xlen={32,64}), and
# each `make install` populates its own install tree containing the
# generated VX_types.h and the matching libvortex2.a. Point
# VORTEX_PATH_32 / VORTEX_PATH_64 at those install trees; each
# resolves to <path>/kernel/include for headers.

# Build 32-bit kernel bitcode if VORTEX_PATH_32 is set and valid.
if(DEFINED VORTEX_PATH_32 AND EXISTS "${VORTEX_PATH_32}/kernel/include")
    set(CLANG_FLAGS "-ffreestanding" "-target" "${LLVM_TARGET_32}" "-march=${TARGET_MARCH_32}" "-emit-llvm" "-DPOCL_DEVICE_ADDRESS_BITS=32" "-I${VORTEX_PATH_32}/kernel/include")
    make_kernel_bc(KERNEL_BC_32 "${LLVM_TARGET_32}" "VX32-BCs" 0 0 0 ${KERNEL_SOURCES})
else()
    message(STATUS "Skipping Vortex riscv32 kernel build: set -DVORTEX_PATH_32=<rv32-install-root> to enable")
endif()

# Build 64-bit kernel bitcode if VORTEX_PATH_64 is set and valid.
if(DEFINED VORTEX_PATH_64 AND EXISTS "${VORTEX_PATH_64}/kernel/include")
    set(CLANG_FLAGS "-ffreestanding" "-target" "${LLVM_TARGET_64}" "-march=${TARGET_MARCH_64}" "-emit-llvm" "-DPOCL_DEVICE_ADDRESS_BITS=64" "-I${VORTEX_PATH_64}/kernel/include")
    make_kernel_bc(KERNEL_BC_64 "${LLVM_TARGET_64}" "VX64-BCs" 0 0 0 ${KERNEL_SOURCES})
else()
    message(STATUS "Skipping Vortex riscv64 kernel build: set -DVORTEX_PATH_64=<rv64-install-root> to enable")
endif()

set(KERNEL_BC_INSTALL_LIST)

if(KERNEL_BC_32)
    message(STATUS "${LLVM_TARGET_32} Kernel BC: ${KERNEL_BC_32}")
    list(APPEND KERNEL_BC_LIST "${KERNEL_BC_32}")
    add_custom_target("kernel_${LLVM_TARGET_32}" DEPENDS ${KERNEL_BC_32})
    list(APPEND KERNEL_TARGET_LIST "kernel_${LLVM_TARGET_32}")
    list(APPEND KERNEL_BC_INSTALL_LIST "${KERNEL_BC_32}")
endif()

if(KERNEL_BC_64)
    message(STATUS "${LLVM_TARGET_64} Kernel BC: ${KERNEL_BC_64}")
    list(APPEND KERNEL_BC_LIST "${KERNEL_BC_64}")
    add_custom_target("kernel_${LLVM_TARGET_64}" DEPENDS ${KERNEL_BC_64})
    list(APPEND KERNEL_TARGET_LIST "kernel_${LLVM_TARGET_64}")
    list(APPEND KERNEL_BC_INSTALL_LIST "${KERNEL_BC_64}")
endif()

set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

if(KERNEL_BC_INSTALL_LIST)
    install(FILES ${KERNEL_BC_INSTALL_LIST} DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}" COMPONENT "lib")
endif()
