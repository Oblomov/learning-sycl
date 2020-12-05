CXX ?= compute++

SYCL_PATH != dirname $$(which $(CXX))
CXX_Model != $(CXX) --version | head -n1 | cut -f2 -d\ 

CXXFLAGS_ComputeCpp = -sycl-driver
CXXFLAGS_oneAPI = -fsycl

CXXFLAGS ?=
CXXFLAGS += -g -Wall
CXXFLAGS += $(CXXFLAGS_${CXX_Model})

CPPFLAGS ?=
CPPFLAGS += -I$(SYCL_PATH)/../include
CPPFLAGS += -I$(SYCL_PATH)/../include/sycl

LDFLAGS ?=
LDFLAGS += -L$(SYCL_PATH)/../lib
LDFLAGS += -Wl,-rpath=$(SYCL_PATH)/../lib

LDLIBS_ComputeCpp = -lComputeCpp

LDLIBS ?=
LDLIBS += $(LDLIBS_${CXX_Model})
LDLIBS += -lOpenCL



TARGETS = sample sample-select sample-reduce

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: clean all

