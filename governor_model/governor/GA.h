//
// Created by ocelot on 1/29/24.
//

#ifndef EMBEDDEDSYSTEMS_GA_H
#define EMBEDDEDSYSTEMS_GA_H

#include <stdio.h>

#define GA_LATENCY_PENALTY     = 200
#define GA_FPS_PENALTY         = 200
#define GA_POPULATION_SIZE     = 100
#define GA_SELECTION_PRESSURE  = 1.0
#define GA_LAYER_MUTATE_CHANCE = 70

const int NETWORK_SIZE = 8;

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
} Chromosome;

Chromosome genetic_algorithm(int population_size,
                             int target_latency,
                             int target_fps,
                             int staleness_limit
                            );

void chromosomeToString(const Chromosome* c, char* buffer, size_t bufferSize);
#endif  // EMBEDDEDSYSTEMS_GA_H
