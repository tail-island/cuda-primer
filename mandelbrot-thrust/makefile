NVCC      = nvcc
NVCCFLAGS = -std=c++17 -arch=sm_61 --expt-relaxed-constexpr
CXXFLAGS  = -std=c++17 -O3 -Wall -march=native
LDFLAGS   = -L/opt/cuda/targets/x86_64-linux/lib/ -lcudart

PROGRAM   = mandelbrot-thrust
SRCS      = $(wildcard *.cu)
CPPS      = $(SRCS:.cu=.cpp)
OBJS      = $(CPPS:.cpp=.o)
DEPS      = $(SRCS:.cu=.d)

$(PROGRAM): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

-include $(DEPS)

$(OBJS): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(CPPS): %.cpp: %.cu
	$(NVCC) $(NVCCFLAGS) -cuda -MMD -o $@ $<

clean:
	$(RM) $(PROGRAM) $(CPPS) $(OBJS) $(DEPS)
