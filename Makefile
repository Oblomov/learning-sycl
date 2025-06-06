# Known SYCL compilers. Override with `make CXX=/path/to/compiler`
CXX := compute++ icpx dpcpp acpp
CXX_ := icpx

CXX != for s in ${CXX} ; do command -v $$s && break ; done

CXX := $(CXX)$(CXX_$(CXX))

SYCL_PATH != dirname $$(command -v $(CXX) || echo .)
CXX_Model != $(CXX) --acpp-version > /dev/null 2>&1 && echo acpp || $(CXX) --version | head -n1 | cut -f2 -d\ 

CXXFLAGS_ComputeCpp = -sycl-driver
CXXFLAGS_oneAPI = -fsycl
CXXFLAGS_clang = -fsycl

CPPFLAGS_acpp = -I$(SYCL_PATH)/../include/AdaptiveCpp

CXXFLAGS ?=
CXXFLAGS += -g -Wall -O3
CXXFLAGS += $(CXXFLAGS_${CXX_Model})

CPPFLAGS ?=
CPPFLAGS += -I$(SYCL_PATH)/../include
CPPFLAGS += -I$(SYCL_PATH)/../include/sycl

CPPFLAGS += $(CPPFLAGS_${CXX_Model})

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

show:
	@printf 'CXX=%s\n' "$(CXX)"
	@printf 'CXX_Model=%s\n' "$(CXX_Model)"
	@printf 'CPPFLAGS=%s\n' "$(CPPFLAGS)"
	@printf 'CXXFLAGS=%s\n' "$(CXXFLAGS)"
	@printf 'LDFLAGS=%s\n' "$(LDFLAGS)"
	@printf 'LDLIBS=%s\n' "$(LDLIBS)"

.PHONY: clean all show

