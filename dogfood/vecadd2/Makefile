LLVM_HOME ?= /usr/lib/llvm-10
SYSROOT ?= ~/dev/riscv-gnu-toolchain/drops/riscv32-unknown-elf
POCL_CC_PATH ?= $(wildcard ~/dev/pocl/drops_vortex_cc)
POCL_RT_PATH ?= $(wildcard ~/dev/pocl/drops_vortex_rt)
VORTEX_PATH ?= $(wildcard ~/dev/vortex_live/driver/sw/rtlsim)
VORTEX_RT_PATH ?= $(wildcard ~/dev/vortex_live/runtime)

CXXFLAGS += -std=c++11 -O0 -g -Wall -fpermissive -Wextra -pedantic -Wfatal-errors

CXXFLAGS += -I$(POCL_RT_PATH)/include

LDFLAGS += -L$(POCL_RT_PATH)/lib -L$(VORTEX_PATH) -lOpenCL -lvortex

PROJECT=vecadd

SRCS = main.cc

all: $(PROJECT) kernel.pocl

kernel.pocl: kernel.cl
	SYSROOT=$(SYSROOT) VORTEX_RUNTIME_PATH=$(VORTEX_RT_PATH) POCL_DEBUG=all LD_LIBRARY_PATH=$(POCL_CC_PATH)/lib:$(VORTEX_PATH) $(POCL_CC_PATH)/bin/poclcc -o kernel.pocl kernel.cl
 
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