#include "perf_predictor.h"
#include "GA.h"
#include "fitness.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;

const double levels[] = {
    500000, 667000, 1000000, 1200000, 1398000, 1512000, 1608000,
    1704000, 1800000, 1908000, 2016000, 2100000, 2208000
};

void pp2b(int pp1, int pp2, int stage, double* outputs) {
    int steps[4] = {0, pp1, pp2, LAYERS};
    for (int i = steps[stage]; i < steps[stage+1]; i++) {
        outputs[i] = 1.0;
    }
}

double accumulate_sum_aux(double a, double b) {
    return a + b*2;
}

double sigmoid(double n) {
    return (1 / (1 + exp(-n)));
}

/* Note: in-place. */
void sigmoidv(matrix m) {
    for (size_t i; i < m.rows; i++) {
        for (size_t j; j < m.rows; j++) {
            m.data[i][j] = sigmoid(m.data[i][j]);
        }
    }
}

double model(matrix m, matrix* params) {
    assert(m.rows == 1);
    assert(m.cols == 8);

    matmul_inplace(m, params[0]);
    matadd_inplace(m, params[1]);
    sigmoidv(m);
    matmul_inplace(m, params[2]);
    matadd_inplace(m, params[3]);
    double res = m.data[0][0];
    matrix_destroy(m);

    return res;
}

void predict_performance(chromosome* chromosome_, matrix* params,
                         double* output_latency, double* util) {
    /* Get chromosome data. */
    int pp1 = chromosome_->genes[0]->layers;
    int pp2 = pp1 + chromosome_->genes[1]->layers;
    double bfreq = levels[chromosome_->genes[0]->frequency_level];
    double lfreq = levels[chromosome_->genes[2]->frequency_level];

    /* Input setup. */
    matrix inputs_b = matrix_init(1,8);
    matrix inputs_g = matrix_init(1,8);
    matrix inputs_l = matrix_init(1,8);
    pp2b(pp1, pp2, 0, inputs_b.data[0]);
    pp2b(pp1, pp2, 1, inputs_g.data[0]);
    pp2b(pp1, pp2, 2, inputs_l.data[0]);
    inputs_b.data[0][7] = GHZ/bfreq;
    inputs_g.data[0][7] = GHZ/bfreq;
    inputs_l.data[0][7] = GHZ/lfreq;

    /* Run models. */
    double inf_lat1 = model(inputs_b, &params[0]);
    double inf_lat2 = model(inputs_g, &params[2]);
    double inf_lat3 = model(inputs_l, &params[4]);

    /* Calculate total and max latencies*/
    vector<double> active_lat;
    active_lat.push_back(inf_lat1);

    if (pp2-pp1 > 0) {
        active_lat.push_back(inf_lat2);
    }
    if (pp2 < 8) {
        active_lat.push_back(inf_lat3);
    }
    double partial_sums[3];
    partial_sum(active_lat.begin(), active_lat.end(), partial_sums);
    double total_lat = accumulate(partial_sums, &partial_sums[3], 0.0);
    if (pp2 < 8 and pp2-pp1 == 0) {
        total_lat -= max(0.0, inf_lat1-inf_lat3);
    }

    double max_lat = max(max(inf_lat1, inf_lat2), inf_lat3);

    /* Return values. */
    output_latency[0] = total_lat;
    output_latency[1] = max_lat;
    util[0] = inf_lat1/max_lat;
    util[1] = inf_lat2/max_lat;
    util[2] = inf_lat3/max_lat;
}
