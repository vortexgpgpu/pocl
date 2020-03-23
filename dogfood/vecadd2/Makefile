RISCV_TOOL_PATH ?= $(wildcard ~/dev/riscv-gnu-toolchain/drops)
POCLCC_PATH ?= $(wildcard ~/dev/pocl/drops_vortex_cc)
POCLRT_PATH ?= $(wildcard ~/dev/pocl/drops_vortex_rt)
VORTEX_PATH ?= $(wildcard ~/dev/vortex_opae/driver/sw/simx)

CFLAGS += -O0 -g -Wall -Wextra -pedantic -Wfatal-errors

CFLAGS += -I$(POCLRT_PATH)/include

LDFLAGS += -L$(POCLRT_PATH)/lib -L$(VORTEX_PATH) -lOpenCL -lvortex

PROJECT=vecadd

SRCS = main.c

all: $(PROJECT)

kernel.pocl: kernel.cl
	POCL_DEBUG=all POCL_DEBUG_LLVM_PASSES=1 LD_LIBRARY_PATH=$(RISCV_TOOL_PATH)/lib:$(POCLCC_PATH)/lib:$(VORTEX_PATH) $(POCLCC_PATH)/bin/poclcc -o kernel.pocl kernel.cl
 
$(PROJECT): $(SRCS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

run: $(PROJECT)    
	LD_LIBRARY_PATH=$(POCLRT_PATH)/lib:$(VORTEX_PATH):$(LD_LIBRARY_PATH) ./$(PROJECT) -k kernel.pocl

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > .depend;

clean:
	rm -rf $(PROJECT) *.o *.dump .depend 

ifneq ($(MAKECMDGOALS),clean)
    -include .depend
endif