//
// Created by ocelot on 1/29/24.
//

#include <algorithm>
#include "fitness.h"
#include "GA.h"
#include "perf_predictor.h"
#include "power_predictor.h"
#include "matrix.h"


double penalty(double current, double target, double penalty_c) {
    return std::max(0.0, penalty_c * (current - target));
}


double objective(double latency, double max_lat, double power,
                 double target_l, double target_f,
                 double penalty_l, double penalty_f) {
    return (
        power
        + penalty(latency, target_l, penalty_l)
        + penalty(max_lat, target_f, penalty_f)
    );
}


double fitness(chromosome* chromosome_,
               double target_l, double target_f,
               double penalty_f_c, double penalty_l_c,
               matrix* model_params) {
    double total_max_lat[2] = {};
    double util[3] = {};
    predict_performance(chromosome_, model_params, total_max_lat, util);
    double power = predict_power(chromosome_, util);

    double res = objective(total_max_lat[0], total_max_lat[1], power,
                           target_l, target_f, penalty_l_c, penalty_f_c);

    // cout << total_max_lat[0] << "  " << 1000.0/total_max_lat[1] << endl;
    chromosome_->est_lat = total_max_lat[0];
    chromosome_->est_fps = 1000.0/total_max_lat[1];
    chromosome_->est_pwr = power;
    return 1000.0/res;
}
