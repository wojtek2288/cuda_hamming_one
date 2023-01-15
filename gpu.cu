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

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

float solveWithGpu()
{
    return 0;
}
