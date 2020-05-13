CC  = $(RISCV_GNU_TOOLS_PATH)/bin/riscv32-unknown-elf-gcc
CXX = $(RISCV_GNU_TOOLS_PATH)/bin/riscv32-unknown-elf-g++
DMP = $(RISCV_GNU_TOOLS_PATH)/bin/riscv32-unknown-elf-objdump
HEX = $(RISCV_GNU_TOOLS_PATH)/bin/riscv32-unknown-elf-objcopy
GDB = $(RISCV_GNU_TOOLS_PATH)/bin/riscv32-unknown-elf-gdb

CXXFLAGS = -std=c++11 -g -O0 -march=rv32im -mabi=ilp32 
CXXFLAGS += -ffreestanding # program may not begin at main()
CXXFLAGS += -Wl,--gc-sections # enable garbage collection of unused input sections
CXXFLAGS += -fno-rtti -fno-non-call-exceptions # disable RTTI and exceptions
CXXFLAGS += -I$(POCL_RT_PATH)/include

LIBS = -Wl,--whole-archive lib$(PROJECT).a -Wl,--no-whole-archive $(POCL_RT_PATH)/lib/static/libOpenCL.a

PROJECT = vecadd

SRCS = main.cc

all: $(PROJECT).dump $(PROJECT).elf

lib$(PROJECT).a: kernel.cl
	POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=$(RISCV_GNU_TOOLS_PATH)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -o lib$(PROJECT).a kernel.cl

$(PROJECT).elf: $(SRCS) lib$(PROJECT).a
	$(CXX) $(CXXFLAGS) $(SRCS) $(LIBS) -o $(PROJECT).elf

$(PROJECT).dump: $(PROJECT).elf
	$(DMP) -D $(PROJECT).elf > $(PROJECT).dump

run: $(PROJECT).elf
	POCL_DEBUG=all $(RISCV_GNU_TOOLS_PATH)/bin/qemu-riscv32 $(PROJECT).elf

gdb-s: $(PROJECT).elf
	POCL_DEBUG=all $(RISCV_GNU_TOOLS_PATH)/bin/qemu-riscv32 -g 1234 -d in_asm -D debug.log $(PROJECT).elf

gdb-c: $(PROJECT).elf
	$(GDB) $(PROJECT).elf

clean:
	rm -rf *.a *.o *.elf *.dump ~/.cache/pocl