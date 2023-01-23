all: hamming-one generator

hamming-one: 
	nvcc -o hamming-one main.cu gpu.cu cpu.cpp -lcudart

generator: 
	nvcc -o generator generator.cpp

.PHONY: clean all

clean:
	rm hamming-one generator
