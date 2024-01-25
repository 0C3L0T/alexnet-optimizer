#include <stdio.h>  /* printf */
#include <stdlib.h> /* system, NULL, EXIT_FAILURE, srand, rand */
#include <signal.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include "../parse_results.h"
using namespace std;
#define NETWORK_SIZE 8

static volatile int keepRunning = 1;

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
    char Run_Command[256];
    sprintf(Run_Command, "./graph_alexnet_all_pipe_sync --threads=4  --threads2=2 --n=60 --total_cores=6 --partition_point=%d --partition_point2=%d --order=%s > output.txt", partition_point, partition_point2, order.c_str());
    system(Run_Command);
}

void printFreq(int littleFreq, int bigFreq, int gpuOn) {
    cout << "Little Frequency: " << to_string(littleFreq) << ", ";
    cout << "Big Frequency: " << to_string(bigFreq) << ", ";
    cout << "GPU On: " << to_string(gpuOn) << ", ";
}

void intHandler(int dummy) {
    keepRunning = 0;
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

    srand(time(NULL));
    string order = "B-G-L";
    signal(SIGINT, intHandler);
    signal(SIGTERM, intHandler);
    /* random data generation */
    while (keepRunning) {
        int ran1 = rand() % NETWORK_SIZE + 1;
        int ran2 = rand() % NETWORK_SIZE + 1;
        int rans[] = {ran1, ran2};
        int lowest = ran1 > ran2;
        int pp1 = rans[lowest];
        int pp2 = rans[!lowest];
        int lfreqlvl = rand() % 9;
        int bfreqlvl = rand() % 13;

        initLittleWithFreq(lfreqlvl);
        initBigWithFreq(bfreqlvl);
        runCNN(pp1, pp2, order);
        printFreq(LittleFrequencyTable[lfreqlvl], BigFrequencyTable[bfreqlvl], (pp2!=pp1));
        ParseResults();
    }

    cout << "ending program" << endl;
}