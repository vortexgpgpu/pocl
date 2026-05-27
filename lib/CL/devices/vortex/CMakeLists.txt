#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014-2021 pocl developers
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

unset(CMAKE_CXX_STANDARD)
unset(CMAKE_CXX_STANDARD_REQUIRED)
set_source_files_properties(vortex_utils.cc vortex_utils.h PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS}")

# Vortex SDK is consumed through its install tree ($VORTEX_PATH) via
# pkg-config; the PkgConfig::VORTEX IMPORTED target is created at the
# top-level CMakeLists.txt under `if(ENABLE_VORTEX)` so the target is
# visible across the source tree (this dir, lib/CL/, examples/).
include_directories(${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)
set(CMAKE_CXX_STANDARD ${POCL_CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_program(LLVM_OBJDUMP "llvm-objdump" HINTS "${LLVM_BINDIR}" "${LLVM_PREFIX_BIN}")
if(NOT LLVM_OBJDUMP)
    set(LLVM_OBJDUMP "${LLVM_PREFIX}/bin/llvm-objdump")
endif()

add_pocl_device_library("pocl-devices-vortex" pocl-vortex.c pocl-vortex.h vortex_utils.cc vortex_utils.h)

# vortex_utils.cc references LLVM_PREFIX and LLVM_OBJDUMP as preprocessor
# macros; expose them via compile definitions instead of patching upstream
# cmake/LLVM.cmake.
target_compile_definitions(pocl-devices-vortex PRIVATE
    LLVM_PREFIX="${LLVM_PREFIX}"
    LLVM_OBJDUMP="${LLVM_OBJDUMP}"
)

target_link_libraries(pocl-devices-vortex PRIVATE PkgConfig::VORTEX)
if(ENABLE_LOADABLE_DRIVERS)
  target_link_libraries(pocl-devices-vortex PRIVATE ${PTHREAD_LIBRARY})
endif()

install(FILES "kernel_main.c" "kernel_args.h"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/vortex" COMPONENT "lib")
