//
// Created by ocelot on 1/29/24.
//

#ifndef EMBEDDEDSYSTEMS_FITNESS_H
#define EMBEDDEDSYSTEMS_FITNESS_H
#include "GA.h"
#include "matrix.h"

double fitness(chromosome* chromosome_,
               double target_l, double target_f,
               double penalty_f_c, double penalty_l_c,
               matrix* model_params);

#endif  // EMBEDDEDSYSTEMS_FITNESS_H
