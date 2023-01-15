#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "device_launch_parameters.h"
#include "cpu.h"
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

    auto cpuPairs = solveWithCpu(bitSequences);
    ofstream output("output.txt");

    for (auto pair : cpuPairs)
    {
        cout << pair.first << endl;
        cout << pair.second << endl;
        output << pair.first << endl;
        output << pair.second << endl;
    }
    output.close();
}