#include <stdio.h>  /* printf */
#include <stdlib.h> /* system, NULL, EXIT_FAILURE */
#include <iostream>
#include <fstream>
#include <sstream>
#include "parse_results.h"
using namespace std;

int LittleFrequencyTable[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000};
int BigFrequencyTable[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000, 2100000, 2208000};

int main(int argc, char *argv[])
{
    string Command = "";

    /* Checking if processor is available */
    if (system(NULL))
        puts("Ok");
    else
        exit(EXIT_FAILURE);

    /* Export OpenCL library path */
    system("export LD_LIBRARY_PATH=/data/local/Working_dir");
    setenv("LD_LIBRARY_PATH", "/data/local/Working_dir", 1);

    /* Setup Performance Governor (CPU) */
    system("echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor");
    system("echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor");

    /* Initialize Little and Big CPU with Lowest Frequency */
    Command = "echo " + to_string(LittleFrequencyTable[0]) + " > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq";
    system(Command.c_str());
    Command = "echo " + to_string(BigFrequencyTable[0]) + " > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq";
    system(Command.c_str());

    /* Run everything on Little, Big, and GPU seperately. */

    char Run_Command[150];
    std::array<std::string, 3> orders = {"L-G-B", "B-L-G", "G-B-L"};
    for(const auto& order : orders) {
        sprintf(Run_Command, "./graph_alexnet_all_pipe_sync --threads=4  --threads2=2 --n=60 --total_cores=6 --partition_point=8 --partition_point2=8 --order=%s > output.txt", order);
        system(Run_Command);
        ParseResults();
    }

}