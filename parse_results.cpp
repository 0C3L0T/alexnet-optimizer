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
#include "parse_results.h"
using namespace std;

bool LatencyCondition = 0;
bool FPSCondition = 0;


float StageOneInferenceTime = 0;
float StageTwoInferenceTime = 0;
float StageThreeInferenceTime = 0;

int Target_FPS = 0;
int Target_Latency = 0;

/* Get feedback by parsing the results */
void ParseResults()
{
    float FPS;
    float Latency;
    ifstream myfile("/data/local/Working_dir/output.txt");
    cout << endl;
    /* Read Output.txt File and Extract Data */
    for (std::string line; getline(myfile, line);)
    {
        string temp;
        /* Extract Frame Rate */
        if (line.find("Frame rate is:") == 0)
        {
            // cout<<"line is: "<<line<<std::endl;
            std::istringstream ss(line);
            while (!ss.eof())
            {
                /* extracting word by word from stream */
                ss >> temp;
                /* Checking the given word is float or not */
                if (stringstream(temp) >> FPS)
                {
                    printf("Throughput is: %f FPS", FPS);
                    cout << endl;
                    break;

                }
                temp = "";
            }
        }
        /* Extract Frame Latency */
        if (line.find("Frame latency is:") == 0)
        {
            // cout<<"line is: "<<line<<std::endl;
            std::istringstream ss(line);
            while (!ss.eof())
            {
                /* extracting word by word from stream */
                ss >> temp;
                /* Checking the given word is float or not */
                if (stringstream(temp) >> Latency)
                {
                    printf("Latency is: %f ms", Latency);
                    cout << endl;
                    break;
                }
                temp = "";
            }
        }
        /* Extract Stage One Inference Time */
        if (line.find("stage1_inference_time:") == 0)
        {
            // cout<<"line is: "<<line<<std::endl;
            std::istringstream ss(line);
            while (!ss.eof())
            {
                /* extracting word by word from stream */
                ss >> temp;
                /* Checking the given word is float or not */
                if (stringstream(temp) >> StageOneInferenceTime)
                {
                    // printf("StageOneInferenceTime is: %f ms\n", StageOneInferenceTime);
                    break;
                }
                temp = "";

            }
        }
        /* Extract Stage Two Inference Time */
        if (line.find("stage2_inference_time:") == 0)
        {
            // cout<<"line is: "<<line<<std::endl;
            std::istringstream ss(line);
            while (!ss.eof())
            {
                /* extracting word by word from stream */
                ss >> temp;
                /* Checking the given word is float or not */
                if (stringstream(temp) >> StageTwoInferenceTime)
                {
                    // printf("StageTwoInferenceTime is: %f ms\n", StageTwoInferenceTime);
                    break;
                }
                temp = "";
            }
        }
        /* Extract Stage Three Inference Time */
        if (line.find("stage3_inference_time:") == 0)
        {
            // cout<<"line is: "<<line<<std::endl;
            std::istringstream ss(line);
            while (!ss.eof())
            {
                /* extracting word by word from stream */
                ss >> temp;
                /* Checking the given word is float or not */
                if (stringstream(temp) >> StageThreeInferenceTime)
                {
                    // printf("StageThreeInferenceTime is: %f ms\n", StageThreeInferenceTime);
                    break;
                }
                temp = "";
            }
        }
    }
    /* Check Throughput and Latency Constraints */
    if (Latency <= Target_Latency)
    {
        LatencyCondition = 1; // Latency requirement was met.
    }
    if (FPS >= Target_FPS)
    {
        FPSCondition = 1; // FPS requirement was met.
    }
}
