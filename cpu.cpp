#include <stdlib.h>
#include <algorithm>
#include <time.h>

#include "cpu.h"
#include "defines.h"

#include <iostream>
#include <vector>
using namespace std;

vector<pair<string, string>> solveWithCpu(vector<string> bitSequences)
{
    vector<pair<string, string>> pairs;
    for (int i = 0; i < bitSequences.size(); i++)
    {
        for (int j = i + 1; j < bitSequences.size(); j++)
        {
            int distance = 0;
            for (int k = 0; k < bitSequences[i].length(); k++)
            {
                if (bitSequences[i][k] != bitSequences[j][k])
                {
                    distance++;
                }
            }
            if (distance == 1)
            {
                pairs.push_back({bitSequences[i], bitSequences[j]});
            }
        }
    }
    return pairs;
}