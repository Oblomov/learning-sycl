CXX = compute++

CXXFLAGS ?=
CXXFLAGS += -g -Wall
CXXFLAGS += -sycl-driver

SYCL.dir.cmd = dirname $$(which $(CXX))
SYCL_PATH = $(shell $(SYCL.dir.cmd))$(SYCL.dir.cmd:sh)

CPPFLAGS ?=
CPPFLAGS += -I$(SYCL_PATH)/../include

LDFLAGS ?=
LDFLAGS += -L$(SYCL_PATH)/../lib
LDFLAGS += -Wl,-rpath=$(SYCL_PATH)/../lib

LDLIBS ?=
LDLIBS += -lComputeCpp -lOpenCL


TARGETS = sample sample-select sample-reduce

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

.PHONY: clean all

