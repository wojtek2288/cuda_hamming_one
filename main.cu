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
    if (argc != 2)
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

    // cout << "Starting cpu" << endl;
    // auto cpuPairs = solveWithCpu(bitSequences);
    cout << "Starting gpu" << endl;
    auto gpuPairs = solveWithGpu(bitSequences);

    // ofstream cpu_output("output_cpu.txt");
    ofstream gpu_output("output_gpu.txt");

    // for (auto pair : cpuPairs)
    // {
    //     cpu_output << pair.first << endl;
    //     cpu_output << pair.second << endl;
    // }

    for (auto pair : gpuPairs)
    {
        gpu_output << pair.first << endl;
        gpu_output << pair.second << endl;
    }

    // cpu_output.close();
    gpu_output.close();
}