//
// Created by ocelot on 1/29/24.
//
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include "GA.h"
#include "fitness.h"

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


    // optionally: parse feedback (look at exampleGovernor)

    char buffer[256];
    chromosomeToString(&solution, buffer, 256);
    printf("%s", buffer);
}