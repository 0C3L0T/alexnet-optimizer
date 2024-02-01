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

    // cout << "m.rows " << m.rows << " m.cols" << endl;
    // cout << "param0 dims: " << params[0].rows << " " << params[0].cols << endl;
    // cout << "param1 dims: " << params[1].rows << " " << params[1].cols << endl;
    // cout << "param2 dims: " << params[2].rows << " " << params[2].cols << endl;
    // cout << "param3 dims: " << params[3].rows << " " << params[3].cols << endl;
    matrix temp1 = matmul(m, params[0]);
    // cout << "-1 " << endl;
    matadd_inplace(temp1, params[1]);
    // cout << "0 " << endl;
    // matrix_print(m);
    sigmoidv(temp1);
    // matrix_print(m);
    // cout << "1 " << endl;
    matrix temp2 = matmul(temp1, params[2]);
    // cout << "2" << endl;
    matadd_inplace(temp2, params[3]);
    // cout << "3" << endl;
    double res = temp2.data[0][0];
    // cout << "4" << endl;
    matrix_destroy(m);
    // cout << "5" << endl;
    matrix_destroy(temp1);
    matrix_destroy(temp2);

    // cout << "result is " << res << endl;
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

    // cout << "bfreq: " << bfreq << endl;
    // for (int i = 0; i < 8; i++) {
    //     cout << "input b no: " << i << ": " << inputs_b.data[0][i] << endl;
    // }
    /* Run models. */
    double inf_lat1 = model(inputs_b, &params[0]);
    double inf_lat2 = model(inputs_g, &params[4]);
    double inf_lat3 = model(inputs_l, &params[8]);

    cout << "inf_lat1 " << inf_lat1 << endl;
    cout << "inf_lat2 " << inf_lat2 << endl;
    cout << "inf_lat3 " << inf_lat3 << endl;

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
