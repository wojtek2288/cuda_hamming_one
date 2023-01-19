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

__global__ void findHammingDistance(int *d_bitSequences, int *d_output, int vectorCount, int vectorLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vectorCount)
        return;
    for (int i = idx + 1; i < vectorCount; i++)
    {
        int hammingDistance = 0;
        for (int j = 0; j < vectorLength; j++)
        {
            hammingDistance += __popc(d_bitSequences[idx * vectorLength + j] ^ d_bitSequences[i * vectorLength + j]);
        }
        if (hammingDistance == 1)
        {
            atomicAdd(&d_output[idx * vectorCount + i], 2);
        }
    }
}

vector<pair<string, string>> solveWithGpu(vector<string> bitSequences)
{
    vector<pair<string, string>> pairs;

    int vectorCount = bitSequences.size();
    int vectorLength = bitSequences[0].length();

    int *h_bitSequences = new int[vectorCount * vectorLength];
    int *h_output = new int[vectorCount * vectorCount];
    int *d_bitSequences, *d_output;

    for (int i = 0; i < vectorCount; i++)
    {
        for (int j = 0; j < vectorLength; j++)
        {
            h_bitSequences[i * vectorLength + j] = bitSequences[i][j] - '0';
        }
    }

    std::fill_n(h_output, vectorCount * vectorCount, -1);

    gpuErrorCheck(cudaMalloc(&d_bitSequences, vectorCount * vectorLength * sizeof(int)));
    gpuErrorCheck(cudaMalloc(&d_output, vectorCount * vectorCount * sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_bitSequences, h_bitSequences, vectorCount * vectorLength * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(d_output, h_output, vectorCount * vectorCount * sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (vectorCount + blockSize - 1) / blockSize;
    findHammingDistance<<<numBlocks, blockSize>>>(d_bitSequences, d_output, vectorCount, vectorLength);
    gpuErrorCheck(cudaMemcpy(h_output, d_output, vectorCount * vectorCount * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < vectorCount; i++)
    {
        for (int j = 0; j < vectorCount; j++)
        {
            if (h_output[i * vectorCount + j] != -1)
            {
                pairs.push_back({bitSequences[i], bitSequences[j]});
            }
        }
    }

    gpuErrorCheck(cudaFree(d_bitSequences));
    gpuErrorCheck(cudaFree(d_output));
    delete[] h_bitSequences;
    delete[] h_output;

    return pairs;
}