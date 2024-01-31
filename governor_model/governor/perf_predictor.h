#ifndef EMBEDDEDSYSTEMS_PERFPREDICTOR_H
#define EMBEDDEDSYSTEMS_PERFPREDICTOR_H
#include "GA.h"

#define GHZ 1000000
#define LAYERS 8

void predict_performance(chromosome* chromosome_, double* params,
                         double* output_latency, double* util);

#endif  // EMBEDDEDSYSTEMS_PERFPREDICTOR_H
