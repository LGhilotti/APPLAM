ROOT_DIR := .
SRC_DIR := $(ROOT_DIR)/src
SPIKES_DIR := $(SRC_DIR)/spikes
PROTO_DIR := $(ROOT_DIR)/protos
STAN_ROOT_DIR := $(if $(STAN_ROOT_DIR),$(STAN_ROOT_DIR),/home/mario/dev/bayesmix/lib/math)

CXX = g++
CFLAGS = \
	-std=c++14 \
	-MMD \
	-I $(STAN_ROOT_DIR) \
	-I $(STAN_ROOT_DIR)/lib/eigen_*/ \
	-I $(STAN_ROOT_DIR)/lib/boost_*/ \
	-I $(STAN_ROOT_DIR)/lib/sundials_*/include \
	-I $(STAN_ROOT_DIR)/lib/tbb_*/include \
	-I $(PROTO_DIR) \
	-D_REENTRANT -fPIC \
	-O3 -march=native -msse2 -funroll-loops -ftree-vectorize -fopenmp -Wfatal-errors
LDLIBS = \
 	$(shell pkg-config --libs protobuf) -L$(STAN_ROOT_DIR)/lib/tbb \
	-lgsl -lgslcblas -lpthread -ltbb -Wl,-rpath,"$(STAN_ROOT_DIR)/lib/tbb"
LDFLAGS = -D_REENTRANT -O3 -fopenmp

PROTO_SRCS = $(wildcard $(PROTO_DIR)/cpp/*.cpp)
SPIKES_SRCS = $(SPIKES_DIR)/test_binary/test_binary.cpp \
				#$(SPIKES_DIR)/test_ctilde_update/test_ctilde.cpp
				#$(SPIKES_DIR)/test_kappas_reduced/test_kappas.cpp
				#$(SPIKES_DIR)/test_derivatives/test_derivat.cpp
				#$(SPIKES_DIR)/test_valgrind/run_test_p6_d4.cpp
				#$(SPIKES_DIR)/p50_inference_test/test_all_p50_d2_3g_inference.cpp
				#$(SPIKES_DIR)/first_inference_test/inf_multiple_params/debug_all_p200_d2_3g.cpp
				#$(SPIKES_DIR)/first_inference_test/test_lambdablock_p4_d2_3g_inference.cpp
				#$(SPIKES_DIR)/first_inference_test/test_sigmabar_inference.cpp
				#$(SPIKES_DIR)/first_inference_test/test_lambdablock_p5_inference.cpp

OUR_SRCS = $(wildcard $(SRC_DIR)/*.cpp) \
		   $(wildcard $(SRC_DIR)/precs/*.cpp) \
		   $(wildcard $(SRC_DIR)/point_process/*.cpp)

OUR_SRCS := $(filter-out ./src/python_exports.cpp, $(OUR_SRCS))

SRCS = $(PROTO_SRCS) $(OUR_SRCS)
OBJS = $(subst .cpp,.o, $(SRCS))
DEPENDS := $(patsubst %.cpp,%.d,$(SRCS))

SPIKES_EXECS = $(subst .cpp,.out, $(SPIKES_SRCS))
SPIKES_OBJS =  $(subst .cpp,.o, $(SPIKES_SRCS))

all: generate_pybind

generate_pybind: $(OBJS)
	$(CXX) -shared $(CFLAGS) `/usr/bin/python3.8 -m pybind11 --includes` \
		src/python_exports.cpp -o \
		pp_mix_high`/usr/bin/python3.8-config --extension-suffix` \
		$(OBJS) $(LDLIBS) -fopenmp

test: $(SPIKES_EXECS)

$(SPIKES_EXECS): %.out: %.o $(OBJS)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJS) $< $(LDLIBS)

-include $(DEPENDS)

%.o : %.cpp
	$(CXX) $(CFLAGS) `/usr/bin/python3.8 -m pybind11 --includes` \
	-MMD -MP -c $< -o $@

clean:
	rm $(SPIKES_OBJS) $(OBJS) $(DEPENDS)

distclean: clean

compile_protos:
	@ mkdir -p $(PROTO_DIR)/cpp;
	@ mkdir -p $(PROTO_DIR)/py;
	@ for filename in $(PROTO_DIR)/*.proto; do \
		protoc --proto_path=$(PROTO_DIR) --python_out=$(PROTO_DIR)/py/ $$filename; \
		protoc --proto_path=$(PROTO_DIR) --cpp_out=$(PROTO_DIR)/cpp/ $$filename; \
	done
	@ for filename in $(PROTO_DIR)/cpp/*.cc; do \
	    mv -- "$$filename" "$${filename%.cc}.cpp"; \
	done

	touch $(PROTO_DIR)/__init__.py
	touch $(PROTO_DIR)/py/__init__.py

	2to3 --output-dir=$(PROTO_DIR)/py/ -W -n $(PROTO_DIR)/py/
