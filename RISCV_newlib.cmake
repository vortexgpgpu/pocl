# This is an example Toolchain file to cross-compile for ARM/MIPS/other
# boards from x86_64. Copy & modify
#
# Steps:
# 1) Install g++ and gcc cross-compilers
#    (apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf)
# 2) On your board, install libltdl, ocl-icd and libhwloc + their development headers
# 3) copy the entire root filesystem of the board somewhere on your host,
#    then set CMAKE_FIND_ROOT_PATH below to this path
# 4) run cmake like this:
#          cmake -DHOST_DEVICE_BUILD_HASH=<SOME_HASH> -DOCS_AVAILABLE=0
#           -DCMAKE_TOOLCHAIN_FILE=<path-to-this-file>
#           -DLLC_TRIPLE=<your-triple (e.g.arm-gnueabihf-linux-gnu)
#           -DLLC_HOST_CPU=<your-cpu (e.g. armv7a)>
#           <path-to-pocl-source>

SET(CMAKE_SYSTEM_NAME Linux)

SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")

# specify the cross compiler
SET(CMAKE_C_COMPILER $ENV{RISCV_TOOLCHAIN_PATH}/bin/riscv32-unknown-elf-gcc)
SET(CMAKE_CXX_COMPILER $ENV{RISCV_TOOLCHAIN_PATH}/bin/riscv32-unknown-elf-g++)

# should work, but does not yet. Instead set FIND_ROOT below
# set(CMAKE_SYSROOT $ENV{RISCV_TOOLCHAIN_PATH}/riscv32-unknown-elf)

# where is the target environment
SET(CMAKE_FIND_ROOT_PATH  $ENV{RISCV_TOOLCHAIN_PATH}/riscv32-unknown-elf)

# where to find libraries in target environment
SET(CMAKE_LIBRARY_PATH $ENV{RISCV_TOOLCHAIN_PATH}/riscv32-unknown-elf/lib)

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
