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
        clock_t executionStart, executionEnd;

        cout << "Starting cpu" << endl;

        executionStart = clock();
        auto cpuPairs = solveWithCpu(bitSequences);
        executionEnd = clock();

        cout << "Cpu took: " << ((float)(executionEnd - executionStart)) / (CLOCKS_PER_SEC / 1000) << " ms" << endl;
        cout << "Number of pairs: " << cpuPairs << endl;
    }
    else if (string(argv[2]) == "--gpu")
    {
        clock_t executionStart, executionEnd;

        cout << "Starting gpu" << endl;

        executionStart = clock();
        auto gpuPairs = solveWithGpu(bitSequences);
        executionEnd = clock();

        cout << "Gpu took: " << ((float)(executionEnd - executionStart)) / (CLOCKS_PER_SEC / 1000) << " ms" << endl;
        cout << "Number of pairs: " << gpuPairs << endl;
    }
}