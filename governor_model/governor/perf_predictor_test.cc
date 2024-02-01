#include "GA.h"
#include "perf_predictor.h"

int main(void)
{
    double target_fps = 10;
    double target_latency = 200;

    double inv_fps = 1000/target_fps;

    string param_files[] = {
        "s1_1w.txt",
        "s1_1b.txt",
        "s1_2w.txt",
        "s1_2b.txt",
        "s2_1w.txt",
        "s2_1b.txt",
        "s2_2w.txt",
        "s2_2b.txt",
        "s3_1w.txt",
        "s3_1b.txt",
        "s3_2w.txt",
        "s3_2b.txt"
    };

    size_t rows[] = { 8, 1, 8, 1 };
    size_t cols[] = { 8, 8, 1, 1 };

    matrix params[12];
    for (int i = 0; i < 12; i++) {
        params[i] = matrix_from_file("weights/" + param_files[i], rows[i%4], cols[i%4]);
    }
    double latency[2];
    double utils[3];
    chromosome c = create_random_chromosome();
    predict_performance(&c, params, latency, utils);

    cout << "max lat " << latency[0] << endl;

    return 0;
}
