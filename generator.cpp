#include <iostream>
#include <fstream>
#include <cstdlib>

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        return 1;
    }

    int vectorCount = atoi(argv[1]);
    int vectorLength = atoi(argv[2]);
    std::string file = argv[3];
    std::ofstream out(file);

    for (int i = 0; i < vectorCount; i++)
    {
        for (int j = 0; j < vectorLength; j++)
        {
            out << rand() % 2;
        }
        out << std::endl;
    }

    out.close();
    return 0;
}