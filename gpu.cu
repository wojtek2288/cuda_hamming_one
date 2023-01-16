#define gpuErrorCheck(ans)                    \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpu.cuh"
#include "defines.h"
#include "cuda_runtime_api.h"
#include <vector>
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

__global__ void findHammingDistance(int *d_bitSequences, int *d_output, int *d_flag, int vectorCount, int vectorLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vectorCount)
        return;
    for (int i = idx + 1; i < vectorCount; i++)
    {
        int hammingDistance = 0;
        for (int j = 0; j < vectorLength; j++)
        {
            if ((d_bitSequences[idx * vectorLength + j] != d_bitSequences[i * vectorLength + j]))
            {
                hammingDistance++;
            }
        }
        if (hammingDistance == 1)
        {
            int pairIdx = min(idx, i);
            if (!d_flag[pairIdx])
            {
                atomicCAS(&d_output[pairIdx], -1, max(idx, i));
                d_flag[pairIdx] = 1;
            }
        }
    }
}

vector<pair<string, string>> solveWithGpu(vector<string> bitSequences)
{
    vector<pair<string, string>> pairs;

    int vectorCount = bitSequences.size();
    int vectorLength = bitSequences[0].length();

    int *h_bitSequences = new int[vectorCount * vectorLength];
    int *h_output = new int[vectorCount];
    int *h_flag = new int[vectorCount];
    int *d_bitSequences, *d_output, *d_flag;

    for (int i = 0; i < vectorCount; i++)
    {
        for (int j = 0; j < vectorLength; j++)
        {
            h_bitSequences[i * vectorLength + j] = bitSequences[i][j] - '0';
        }
        h_output[i] = -1;
        h_flag[i] = 0;
    }

    cudaMalloc(&d_bitSequences, vectorCount * vectorLength * sizeof(int));
    cudaMalloc(&d_output, vectorCount * sizeof(int));
    cudaMalloc(&d_flag, vectorCount * sizeof(int));
    cudaMemcpy(d_bitSequences, h_bitSequences, vectorCount * vectorLength * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, vectorCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, h_flag, vectorCount * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (vectorCount + blockSize - 1) / blockSize;
    findHammingDistance<<<numBlocks, blockSize>>>(d_bitSequences, d_output, d_flag, vectorCount, vectorLength);
    cudaMemcpy(h_output, d_output, vectorCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag, d_flag, vectorCount * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < vectorCount; i++)
    {
        if (h_output[i] != -1)
        {
            pairs.push_back({bitSequences[i], bitSequences[h_output[i]]});
        }
    }

    cudaFree(d_bitSequences);
    cudaFree(d_output);
    cudaFree(d_flag);
    delete[] h_bitSequences;
    delete[] h_output;
    delete[] h_flag;

    return pairs;
}