CXXFLAGS = -std=c++17 -O3 -Wall -march=native
LDFLAGS  =

PROGRAM  = mandelbrot
SRCS     = $(wildcard *.cpp)
OBJS     = $(SRCS:.cpp=.o)
DEPS     = $(SRCS:.cpp=.d)

$(PROGRAM): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

-include $(DEPS)

$(OBJS): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ -MMD $<

clean:
	$(RM) $(PROGRAM) $(OBJS) $(DEPS)