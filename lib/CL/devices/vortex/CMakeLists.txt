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
include_directories(${LLVM_INCLUDE_DIRS} ${VORTEX_PREFIX}/runtime/include ${CMAKE_CURRENT_SOURCE_DIR}/../../../llvmopencl)
set(CMAKE_CXX_STANDARD ${POCL_CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_pocl_device_library("pocl-devices-vortex" pocl-vortex.c pocl-vortex.h vortex_utils.cc vortex_utils.h)

if(ENABLE_LOADABLE_DRIVERS)
  target_link_libraries(pocl-devices-vortex PRIVATE ${VORTEX_PREFIX}/runtime/lib/libvortex.so ${PTHREAD_LIBRARY})
endif()

install(FILES "kernel_main.c" "kernel_args.h"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}/vortex" COMPONENT "lib")