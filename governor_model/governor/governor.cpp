//
// Created by ocelot on 1/29/24.
//
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include "GA.h"
#include "fitness.h"

using namespace std;

int LittleFrequencyTable[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000};
int BigFrequencyTable[] = {500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000, 1704000, 1800000, 1908000, 2016000, 2100000, 2208000};


int main(int argc, char *argv[]) {
    // Display help message
    std::cout << "Usage: " << argv[0] << " <Population_Size> <Target_Latency> <Target_FPS> <Staleness_Limit>\n";

    if (argc != 5) {
        std::cerr << "Error: Insufficient arguments. Please provide values for all parameters.\n";
        return EXIT_FAILURE;
    }

    // Parse population size
    int population_size = std::stoi(argv[1]);

    // Parse target latency
    int target_latency = std::stoi(argv[2]);

    // Parse target FPS
    int target_fps = std::stoi(argv[3]);

    // Parse staleness limit
    int staleness_limit = std::stoi(argv[4]);

    float (*fitness_function) (chromosome*);

    // pass requirements to GA
    chromosome solution = genetic_algorithm(population_size,
                                      target_latency,
                                      target_fps,
                                      staleness_limit,
                                      fitness_function
                                      );

    int big_freq = BigFrequencyTable[solution.genes[0]->frequency_level];
    int little_freq = LittleFrequencyTable[solution.genes[2]->frequency_level];

    int pp1 = solution.genes[0]->layers;
    int pp2 = pp1 + solution.genes[1]->layers;

    /* Export OpenCL library path */
    system("export LD_LIBRARY_PATH=/data/local/Working_dir");
    setenv("LD_LIBRARY_PATH", "/data/local/Working_dir", 1);

    /* Setup Performance Governor (CPU) */
    system("echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor");
    system("echo performance > /sys/devices/system/cpu/cpufreq/policy2/scaling_governor");

    // set little and big frequencies
    string Command = "";
    Command = "echo " + to_string(little_freq) + " > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq";
    system(Command.c_str());
    Command = "echo " + to_string(big_freq) + " > /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq";
    system(Command.c_str());

    // print configuration
    char buffer[256];
    chromosomeToString(&solution, buffer, 256);
    printf("%s", buffer);

    int N_Frames = 20;
    // run graph
    char Run_Command[150];
    sprintf(Run_Command, "./graph_alexnet_pipe_all --threads=4 --threads2=2 --target=NEON --n=%d --partition_point=%d --partition_point2=%d --order=B-G-L > output.txt",
            N_Frames, pp1, pp2);
    system(Run_Command);

    // optionally: parse feedback (look at exampleGovernor)

}