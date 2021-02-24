DIRS = mandelbrot mandelbrot-cuda mandelbrot-thrust monte-carlo monte-carlo-thrust matrix-operation matrix-operation-cuda matrix-operation-cublas

all: $(DIRS)
clean: $(DIRS)

$(DIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: all $(DIRS)
