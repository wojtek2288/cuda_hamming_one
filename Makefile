all: hamming-one generator check

hamming-one: 
	nvcc -o hamming-one main.cu gpu.cu cpu.cpp

generator: 
	nvcc -o generator generator.cpp

check: 
	nvcc -o check check.cpp

.PHONY: clean all

clean:
	rm hamming-one generator check
