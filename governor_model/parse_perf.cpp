/*Instructions to Run
On Your Computer:
    arm-linux-androideabi-clang++ -static-libstdc++ Governor.cpp -o Governor
    adb push Governor /data/local/Working_dir
On the Board:
    chmod +x Governor.sh
    ./Governor graph_alexnet_all_pipe_sync #NumberOFPartitions #TargetFPS #TargetLatency
*/

#include <stdio.h>  /* printf */
#include <stdlib.h> /* system, NULL, EXIT_FAILURE */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include "parse_results.h"
using namespace std;

#define PARSEAMOUNT 2

array<string, PARSEAMOUNT> lineCharecteristics =
{
    "Frame rate is:",
    "Frame latency is:",
};

array<string, PARSEAMOUNT> outputLines =
{
    "fps:",
    "latency:",
};

string getOutput(int index, string text)
{
    float floatOutput;
    if (stringstream(text) >> floatOutput)
    {
        return to_string(floatOutput);
    }
    return "";
}

/* Get feedback by parsing the results */
void ParseResults()
{
    ifstream myfile("/data/local/Working_dir/output.txt");

    /* Read Output.txt File and Extract Data */
    for (std::string line; getline(myfile, line);)
    {
        string temp;
        for(int i = 0; i < PARSEAMOUNT; i++) {
            if (line.find(lineCharecteristics[i]) == 0)
            {
                std::istringstream ss(line);
                while (!ss.eof())
                {
                    /* Extracting word by word from stream */
                    ss >> temp;
                    /* Checking the given word is float or not */
                    string output = getOutput(i, temp);
                    if (output != "")
                    {
                        cout << outputLines[i] << output << ", ";
                        break;
                    }
                    temp = "";
                }
            }
        }
    }
    cout << endl;
}
