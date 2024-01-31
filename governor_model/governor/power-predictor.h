#define PP_BASELINE_AMPS 0.45

#define PP_BF_BASE  0.080
#define PP_LF_BASE  0.055
#define PP_BF_RANGE 0.673
#define PP_LF_RANGE 0.091

#define PP_BIG_GPU_SUPP_FACTOR 0.13

#define PP_GPU_AMP 0.12

#include "GA.h"

double bigamp(double bf);
double smlamp(double lf);
double scalebf(double bf);
double scalelf(double lf);
double totalwatts(double lf, double bf, double butil, double gutil, double lutil);
double predict_power(chromosome chromosome, double l1util, double l2util, double l3util);
