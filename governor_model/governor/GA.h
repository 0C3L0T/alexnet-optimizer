//
// Created by ocelot on 1/29/24.
//

#ifndef EMBEDDEDSYSTEMS_GA_H
#define EMBEDDEDSYSTEMS_GA_H

#include <stdio.h>

enum component_type {
  BIG,
  GPU,
  LITTLE,
};

typedef struct {
  component_type type;
  int            layers;
  int            frequency_level;
} gene;

typedef struct {
  gene* genes[3];
  float fitness;
} chromosome;

chromosome genetic_algorithm(int population_size,
                             int target_latency,
                             int target_fps,
                             int staleness_limit,
                             float (*fitness_function)(chromosome*));

void chromosomeToString(const chromosome* c, char* buffer, size_t bufferSize);
#endif  // EMBEDDEDSYSTEMS_GA_H
