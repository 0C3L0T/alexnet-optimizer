#include "power-predictor.h"

#include <algorithm>
#include <cmath>

const int PP_LEVELS[] = { 500000,  667000,  1000000, 1200000, 1398000, 1512000, 1608000,
                          1704000, 1800000, 1908000, 2016000, 2100000, 2208000 };

double bigamp(double bf) {
  bf         = bf * 1e-3;
  double ans = 2.77904e-10 * pow(bf, 3) - 8.72535e-7 * pow(bf, 2) + 0.00102561 * bf - 0.15;
  return ans;
}

double smlamp(double lf) {
  lf = lf * 1e-3;
  return 7.0e-5 * lf + 0.02;
}

double scalebf(double bf) {
  return (bigamp(bf) - PP_BF_BASE) / PP_BF_RANGE;
}

double scalelf(double lf) {
  return (smlamp(lf) - PP_LF_BASE) / PP_LF_RANGE;
}

double totalwatts(double lf, double bf, double butil, double gutil, double lutil) {
  return (PP_BASELINE_AMPS + smlamp(lf) * lutil + bigamp(bf) * butil
          + PP_BIG_GPU_SUPP_FACTOR * scalebf(bf) * std::max(0.0, gutil - butil) + PP_GPU_AMP * gutil)
         * 5.0;
}

double predict_power(chromosome* chr, double l1util, double l2util, double l3util) {
  double bfreq = PP_LEVELS[chr->genes[0]->frequency_level];
  double lfreq = PP_LEVELS[chr->genes[2]->frequency_level];
  return totalwatts(lfreq, bfreq, l1util, l2util, l3util);
}
