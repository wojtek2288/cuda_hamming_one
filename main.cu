#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cpu.h"
#include "gpu.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        cout << "Wrong number of arguments" << endl;
        return 1;
    }

    string filename = argv[1];
    vector<string> bitSequences;
    ifstream input(filename);
    std::string line;

    while (getline(input, line))
    {
        bitSequences.push_back(line);
    }
    input.close();

    if (string(argv[2]) == "--cpu")
    {
        ofstream cpu_output("output_cpu.txt");
        clock_t executionStart, executionEnd;

        cout << "Starting cpu" << endl;

        executionStart = clock();
        auto cpuPairs = solveWithCpu(bitSequences);
        executionEnd = clock();

        cout << "Cpu took: " << ((float)(executionEnd - executionStart)) / (CLOCKS_PER_SEC / 1000) << " ms" << endl;

        for (auto pair : cpuPairs)
        {
            cpu_output << pair.first << "," << pair.second << endl;
        }

        cpu_output.close();
    }
    else if (string(argv[2]) == "--gpu")
    {
        ofstream gpu_output("output_gpu.txt");
        clock_t executionStart, executionEnd;

        cout << "Starting gpu" << endl;

        executionStart = clock();
        auto gpuPairs = solveWithGpu(bitSequences);
        executionEnd = clock();

        cout << "Gpu took: " << ((float)(executionEnd - executionStart)) / (CLOCKS_PER_SEC / 1000) << " ms" << endl;

        for (auto pair : gpuPairs)
        {
            gpu_output << pair.first << "," << pair.second << endl;
        }

        gpu_output.close();
    }
}