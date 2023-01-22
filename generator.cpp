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

    std::string previousVector = "";

    for (int i = 0; i < vectorCount; i++)
    {
        if (i % 15 == 0 && i > 0)
        {
            int random_bit = rand() % vectorLength;
            previousVector[random_bit] = previousVector[random_bit] == '0' ? '1' : '0';
            out << previousVector << std::endl;
            previousVector = "";
        }
        else
        {
            for (int j = 0; j < vectorLength; j++)
            {
                char current_bit = (rand() % 2) + '0';
                out << current_bit;
                if (i % 14 == 0 && i > 0)
                {
                    previousVector += current_bit;
                }
            }
            out << std::endl;
        }
    }

    out.close();
    return 0;
}