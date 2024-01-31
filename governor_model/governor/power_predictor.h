#ifndef EMBEDDEDSYSTEMS_POWERPREDICTOR_H
#define EMBEDDEDSYSTEMS_POWERPREDICTOR_H
#include "GA.h"

#define BASELINE_AMPS       0.45
#define BF_BASE             0.080  // bigamp(500000)
#define LF_BASE             0.055  // smlamp(500000)
#define BF_RANGE            0.673  // bigamp(2208000)-bigamp(500000)
#define LF_RANGE            0.091  // smlamp(1800000)-smlamp(500000)
#define BIG_GPU_SUPP_FACTOR 0.13
#define GPU_AMP             0.12  // GPU power is constant since we can't control the frequency.

double predict_power(chromosome* chromosome_, double* util);

#endif  // EMBEDDEDSYSTEMS_POWERPREDICTOR_H
