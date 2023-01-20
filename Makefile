# Known SYCL compilers. Override with `make CXX=/path/to/compiler`
CXX := compute++ icpx dpcpp
CXX_ := icpx

CXX != for s in ${CXX} ; do command -v $$s && break ; done

CXX := $(CXX)$(CXX_$(CXX))

SYCL_PATH != dirname $$(command -v $(CXX) || echo .)
CXX_Model != $(CXX) --version | head -n1 | cut -f2 -d\ 

CXXFLAGS_ComputeCpp = -sycl-driver
CXXFLAGS_oneAPI = -fsycl
CXXFLAGS_clang = -fsycl

CXXFLAGS ?=
CXXFLAGS += -g -Wall -O3
CXXFLAGS += $(CXXFLAGS_${CXX_Model})

CPPFLAGS ?=
CPPFLAGS += -I$(SYCL_PATH)/../include
CPPFLAGS += -I$(SYCL_PATH)/../include/sycl
CPPFLAGS += -DSYCL2020_DISABLE_DEPRECATION_WARNINGS

LDFLAGS ?=
LDFLAGS += -L$(SYCL_PATH)/../lib
LDFLAGS += -Wl,-rpath=$(SYCL_PATH)/../lib

LDLIBS_ComputeCpp = -lComputeCpp

LDLIBS ?=
LDLIBS += $(LDLIBS_${CXX_Model})
LDLIBS += -lOpenCL



TARGETS = sample sample-select sample-reduce syclinfo

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: clean all

