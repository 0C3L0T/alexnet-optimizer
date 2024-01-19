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
    system(command.c_str());
}

void initBigWithFreq(int freqTableIdx) {
    string command = "";
    command = "echo " + to_string(BigFrequencyTable[freqTableIdx]) + " > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq";
    system(command.c_str());
}

void runCNN(int partition_point, int partition_point2, const std::string& order) {
    char Run_Command[150];
    sprintf(Run_Command, "./graph_alexnet_all_pipe_sync --threads=4  --threads2=2 --n=60 --total_cores=6 --partition_point=%d --partition_point2=%d --order=%s > output.txt", partition_point, partition_point2, order.c_str());
    system(Run_Command);
}

void printFreq(int littleFreq, int bigFreq, int gpuOn) {
    cout << "Little Frequency: " << to_string(littleFreq) << ", ";
    cout << "Big Frequency: " << to_string(bigFreq) << ", ";
    cout << "GPU On: " << to_string(gpuOn) << ", ";
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

    // Little Cluster at all Frequencies
    for(int i = 0; i < 9; i++) {
        initLittleWithFreq(i);
        runCNN(8, 8, "L-G-B");
        printFreq(LittleFrequencyTable[i], 0, 0);
        ParseResults();
    }

    // Big Cluster at all Frequencies
    for(int i = 0; i < 11; i++) {
        initBigWithFreq(i);
        runCNN(8, 8, "B-L-G");
        printFreq(0, BigFrequencyTable[i], 0);
        ParseResults();
    }

    // GPU (with all Big Frequencies)
    for(int i = 0; i < 11; i++) {
        initBigWithFreq(i);
        runCNN(8, 8, "G-L-B");
        printFreq(0, BigFrequencyTable[i], 1);
        ParseResults();
    }

  /**
  4) Two-partition configuration representing the separation of Convolutional and Fully-Connected Layer, evaluated in all six possible placement combinations at fixed CPU frequencies
  (Big, Little), (Little, Big), (Big, GPU), (GPU, Big), (Little, GPU), (GPU, Little).
  **/

  /**
  For example, the behaviour of Three-Partition configurations needs to be understood.
  Also, the interplay between partitions and frequencies is important.
  Also, there is memory-compute heterogeneity between the layers of the same types due to varying sizes of input, weights, and biases, which is quite interesting.
  Impact of CPU frequency on GPU. You are free to approach this complexity as you see fit.
  **/

//    /* Run everything on Little, Big, and GPU separately. */
//    std::array<std::string, 3> orders = {"L-G-B", "B-L-G", "G-B-L"};
//    for(const auto& order : orders) {
//        runCNN(8, 8, order);
//        ParseResults();
//    }

}