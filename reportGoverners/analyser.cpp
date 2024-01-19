#include <stdio.h>  /* printf */
#include <stdlib.h> /* system, NULL, EXIT_FAILURE */
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include "../parse_results.h"
using namespace std;

int LittleFrequencyTable[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000};
int BigFrequencyTable[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000, 2100000, 2208000};

void setupOpenCL() {
    system("export LD_LIBRARY_PATH=/data/local/Working_dir");
    setenv("LD_LIBRARY_PATH", "/data/local/Working_dir", 1);
}

void setupPerformanceGovernor() {
    system("echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor");
    system("echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor");
}

void enableFan() {
    system("echo 1 > /sys/class/fan/enable");
    system("echo 0 > /sys/class/fan/mode");
    system("echo 4 > /sys/class/fan/level");
}

void initLittleWithFreq(int freqTableIdx) {
    string command = "";
    command = "echo " + to_string(LittleFrequencyTable[freqTableIdx]) + " > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq";
    system(Command.c_str());
}

void initBigWithFreq(int freqTableIdx) {
    string command = "";
    command = "echo " + to_string(BigFrequencyTable[freqTableIdx]) + " > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq";
    system(Command.c_str());
}

int main(int argc, char *argv[])
{
    string Command = "";
    cout << "starting program" << endl;

    /* Checking if processor is available */
    if (system(NULL))
        cout << "Ok" << endl;
    else
        exit(EXIT_FAILURE);

    setupOpenCL();
    setupPerformanceGovernor();
    enableFan();

    /* Initialize Little and Big CPU with Lowest Frequency */
    initLittleWithFreq(0);
    initBigWithFreq(0);

    /* Run everything on Little, Big, and GPU seperately. */
    char Run_Command[150];
    std::array<std::string, 3> orders = {"L-G-B", "B-L-G", "G-B-L"};
    for(const auto& order : orders) {
        sprintf(Run_Command, "./graph_alexnet_all_pipe_sync --threads=4  --threads2=2 --n=60 --total_cores=6 --partition_point=8 --partition_point2=8 --order=G-B-L &> /data/local/Working_dir/output.txt", order.c_str());
        system(Run_Command);
        ParseResults();
    }

}