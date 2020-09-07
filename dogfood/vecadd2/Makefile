LLVM_HOME ?= ~/dev/llvm-project/drops
RISCV_TOOLCHAIN_PATH ?= ~/dev/riscv-gnu-toolchain/drops
SYSROOT ?= $(RISCV_TOOLCHAIN_PATH)/riscv32-unknown-elf

POCL_CC_PATH ?= $(wildcard ~/dev/pocl/drops_vortex_cc)
POCL_RT_PATH ?= $(wildcard ~/dev/pocl/drops_vortex_rt)
VORTEX_RT_PATH ?= $(wildcard ~/dev/vortex_live/runtime)
VORTEX_PATH ?= $(wildcard ~/dev/vortex_live/driver/rtlsim)

CXXFLAGS += -std=c++11 -O0 -g -Wall -fpermissive -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(POCL_RT_PATH)/include

LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_PATH) -lOpenCL -lvortex

PROJECT=vecadd

SRCS = main.cc

all: $(PROJECT) kernel.pocl

kernel.pocl: kernel.cl
	RISCV_TOOLCHAIN_PATH=$(RISCV_TOOLCHAIN_PATH) SYSROOT=$(SYSROOT) LLVM_HOME=$(LLVM_HOME) VORTEX_RT_PATH=$(VORTEX_RT_PATH) POCL_DEBUG=all LD_LIBRARY_PATH=$(LLVM_HOME)/lib:$(POCL_CC_PATH)/lib $(POCL_CC_PATH)/bin/poclcc -o kernel.pocl kernel.cl
 
$(PROJECT): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

run: $(PROJECT) kernel.pocl   
	LD_LIBRARY_PATH=$(POCL_RT_PATH)/lib:$(VORTEX_PATH):$(LD_LIBRARY_PATH) ./$(PROJECT) -k kernel.pocl

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o *.pocl *.dump .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif