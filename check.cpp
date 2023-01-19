#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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

vector<string> split(string phrase, string delimiter)
{
    vector<string> list;
    size_t pos = 0;
    string token;
    while ((pos = phrase.find(delimiter)) != string::npos)
    {
        token = phrase.substr(0, pos);
        list.push_back(token);
        phrase.erase(0, pos + delimiter.length());
    }
    list.push_back(phrase);
    return list;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        return 1;
    }

    ifstream file(argv[1]);
    string line;
    int counter = 0;

    while (getline(file, line))
    {
        std::cout << counter << std::endl;
        auto pair = split(line, ",");

        if (!checkHammingDistance(pair[0], pair[1]))
        {
            cout << "Pair does not have a Hamming distance of 1" << endl;
            return 0;
        }
        counter++;
    }
    cout << "All pairs have a Hamming distance of 1" << endl;
    return 0;
}