#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include "cpu.h"
#include "defines.h"
#include <iostream>
#include <vector>
#include <nmmintrin.h>

using namespace std;

// https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
// https://stackoverflow.com/questions/69278755/linear-index-for-a-diagonal-run-of-an-upper-triangular-matrix?noredirect=1&lq=1
int solveWithCpu(vector<string> bitSequences)
{
    int n = bitSequences.size();
    int len = bitSequences[0].length();
    int pairs = 0;

    for (int k = 0; k < n * (n - 1) / 2; k++)
    {
        unsigned long long int i = n - 2 - floor(sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5);
        unsigned long long int j = k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;

        int hammingDistance = 0;

        for (int l = 0; l < len; l++)
        {
            hammingDistance += _mm_popcnt_u32(bitSequences[i][l] ^ bitSequences[j][l]);
            if (hammingDistance > 1)
            {
                break;
            }
        }

        if (hammingDistance == 1)
        {
            pairs++;
        }
    }
    return pairs;
}