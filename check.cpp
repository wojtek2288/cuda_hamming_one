#include <iostream>
#include <fstream>
#include <string>
using namespace std;

bool checkHammingDistance(string s1, string s2)
{
    int distance = 0;
    if (s1.length() != s2.length())
    {
        return false;
    }

    for (int i = 0; i < s1.length(); i++)
    {
        if (s1[i] != s2[i])
        {
            distance++;
        }
    }

    return distance == 1;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        return 1;
    }

    ifstream file(argv[1]);
    string line1, line2;
    int counter = 0;

    while (getline(file, line1))
    {
        if (counter % 2 == 0)
        {
            line2 = line1;
        }
        else if (!checkHammingDistance(line1, line2))
        {
            cout << "Pair does not have a Hamming distance of 1" << endl;
            return 0;
        }
        counter++;
    }
    cout << "All pairs have a Hamming distance of 1" << endl;
    return 0;
}