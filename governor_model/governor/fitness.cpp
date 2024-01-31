//
// Created by ocelot on 1/29/24.
//

#include <algorithm>
#include "fitness.h"
#include "GA.h"

#include "GA.h"

#define BASELINE_AMPS       0.45
#define BF_BASE             0.080  // bigamp(500000)
#define LF_BASE             0.055  // smlamp(500000)
#define BF_RANGE            0.673  // bigamp(2208000)-bigamp(500000)
#define LF_RANGE            0.091  // smlamp(1800000)-smlamp(500000)
#define BIG_GPU_SUPP_FACTOR 0.13
#define GPU_AMP             0.12  // GPU power is constant since we can't control the frequency.


const double levels[] = {
    500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
    1704000, 1800000, 1908000, 2016000, 2100000, 2208000
};


double bigamp(double bf)
{
    bf = bf*1e-3;
    double ans = (
          2.77904e-10 * bf*bf*bf
        - 8.72535e-7  * bf*bf
        + 0.00102561  * bf
        - 0.15
    );
    return ans;
}

double smlamp(double lf) {
  lf = lf * 1e-3;
  return 7.0e-5 * lf + 0.02;
}

double scalebf(double bf) {
  return (bigamp(bf) - BF_BASE) / BF_RANGE;
}

double scalelf(double lf) {
  return (smlamp(lf) - LF_BASE) / LF_RANGE;
}

double totalwatts(double lf, double bf, double butil, double gutil, double lutil) {
  return (
        BASELINE_AMPS
        + smlamp(lf) * lutil
        + bigamp(bf) * butil
        + BIG_GPU_SUPP_FACTOR * scalebf(bf) * std::max(0.0, gutil - butil)
        + GPU_AMP * gutil
    ) * 5.0;
}


double predict_power(chromosome* chromosome_, double* util) {
    double bfreq = levels[chromosome_->genes[0]->frequency_level];
    double lfreq = levels[chromosome_->genes[2]->frequency_level];
    return totalwatts(lfreq, bfreq, util[0], util[1], util[2]);
}

void predict_performance(chromosome* chromosome_, double* params,
                         double* output_latency, double* util) {
    return;
}


double penalty(double current, double target, double penalty_c)
{
    return std::max(0.0, penalty_c * (current - target));
}


double objective(double latency, double max_lat, double power,
                 double target_l, double target_f,
                 double penalty_l, double penalty_f)
{
    return (
        power
        + penalty(latency, target_l, penalty_l)
        + penalty(max_lat, target_f, penalty_f)
    );
}


double fitness(chromosome* chromosome_,
               double target_l, double target_f,
               double penalty_f_c, double penalty_l_c)
{
    double model_params[3] = {};
    double total_max_lat[2] = {};
    double util[3] = {};
    predict_performance(chromosome_, model_params, total_max_lat, util);
    double power = predict_power(chromosome_, util);

    double res = objective(total_max_lat[0], total_max_lat[1], power,
                           target_l, target_f, penalty_l_c, penalty_f_c);

    chromosome_->est_lat = total_max_lat[0];
    chromosome_->est_fps = 1000.0/total_max_lat[1];
    chromosome_->est_pwr = power;
    return 1000.0/res;
}
