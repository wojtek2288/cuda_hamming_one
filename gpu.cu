#define gpuErrorCheck(ans)                    \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpu.cuh"
#include "defines.h"
#include "cuda_runtime_api.h"
using namespace std;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// https://stackoverflow.com/questions/69278755/linear-index-for-a-diagonal-run-of-an-upper-triangular-matrix?noredirect=1&lq=1
__global__ void findPairs(int *d_bitSequences, int *pairs, int vectorCount, int vectorLength)
{
    long long unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < vectorCount * (vectorCount - 1) / 2)
    {
        long long unsigned int i = vectorCount - 2 - floor(sqrt((float)(4 * vectorCount * (vectorCount - 1) - (8 * k) - 7)) / 2.0 - 0.5);
        long long unsigned int j = k + i + 1 - vectorCount * (vectorCount - 1) / 2 + (vectorCount - i) * ((vectorCount - i) - 1) / 2;
        i = j - i - 1;

        int hammingDistance = 0;

        for (int l = 0; l < vectorLength; l++)
        {
            hammingDistance += __popc(d_bitSequences[i * vectorLength + l] ^ d_bitSequences[j * vectorLength + l]);
            if (hammingDistance > 1)
            {
                break;
            }
        }

        if (hammingDistance == 1)
        {
            atomicAdd(pairs, 1);
        }
    }
}

vector<pair<string, string>> solveWithGpu(vector<string> bitSequences)
{
    vector<pair<string, string>> pairs;

    int vectorCount = bitSequences.size();
    int vectorLength = bitSequences[0].length();

    int *h_bitSequences = new int[vectorCount * vectorLength];
    int *h_pairs;
    int *d_bitSequences, *d_pairs;

    for (int i = 0; i < vectorCount; i++)
    {
        for (int j = 0; j < vectorLength; j++)
        {
            h_bitSequences[i * vectorLength + j] = bitSequences[i][j] - '0';
        }
    }

    gpuErrorCheck(cudaMalloc(&d_bitSequences, vectorCount * vectorLength * sizeof(int)));
    gpuErrorCheck(cudaMalloc(&d_pairs, sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_bitSequences, h_bitSequences, vectorCount * vectorLength * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemset(d_pairs, 0, sizeof(int)));

    int threadCount = 512;
    long long unsigned int n = vectorCount * (vectorCount - 1) / 2;
    int blockCount = (n + threadCount - 1) / threadCount + 1;

    findPairs<<<blockCount, threadCount>>>(d_bitSequences, d_pairs, vectorCount, vectorLength);

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaMemcpy(h_pairs, d_pairs, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaFree(d_pairs));
    gpuErrorCheck(cudaFree(d_bitSequences));

    std::cout << "Number of pairs: " << *h_pairs << std::endl;

    delete[] h_bitSequences;

    return pairs;
}