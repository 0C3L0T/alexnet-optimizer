#include "GA.h"
#include "power_predictor.h"
#include <algorithm>


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
