HEADERS=cmdoptions.hpp ../common/basic.hpp ../common/cmdparser.hpp ../common/oclobject.hpp
SOURCES=cmdoptions.cpp gemm.cpp ../common/basic.cpp ../common/cmdparser.cpp ../common/oclobject.cpp

ifeq ($(CONFIG),debug)
	OPT =-O0 -g
else
	OPT =
endif

all: gemm

gemm: $(HEADERS) $(SOURCES) Makefile
	g++ $(SOURCES) -I../common -lOpenCL -ogemm -std=gnu++0x $(OPT)

clean:
	rm -f gemm

