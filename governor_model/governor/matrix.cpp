#include "matrix.h"
#include <stdlib.h>
#include <assert.h>


void matrix_destroy(matrix m) {
    free(m.data);
}

matrix matrix_init(unsigned rows, unsigned cols) {
    double** data = (double**) calloc(rows * cols, sizeof(double));

    matrix m = {
        data,
        rows,
        cols
    };
    return m;
}

void set_row(matrix m, int row, double* data) {
    for (int i = 0; i < m.cols; i++) {
        m.data[row][i] = data[i];
    }
}

void set_col(matrix m, int col, double* data) {
    for (int i = 0; i < m.cols; i++) {
        m.data[i][col] = data[i];
    }
}

matrix matmul(matrix m1, matrix m2) {

    assert(m1.cols == m2.rows);
    matrix res = matrix_init(m1.rows, m2.cols);

    for (int i = 0; i < res.rows; i++) {
        for (int j = 0; j < res.cols; j++) {
            for (int k = 0; k < m2.rows; k++) {
                res.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }

    return res;
}


