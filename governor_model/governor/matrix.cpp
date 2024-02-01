#include "matrix.h"
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;


void matrix_destroy(matrix m) {
    free(m.data);
    free(m.core);
}

matrix matrix_from_file(string filename, size_t rows, size_t cols) {
    ifstream f(filename);
    matrix m = matrix_init(8, 8);
    /* Read Output.txt File and Extract Data */
    size_t i = 0;
    for (string line; getline(f, line) && i < 8; i++) {
        istringstream ss(line);
        for (size_t j = 0; !ss.eof() && ss.peek()!='\n'; j++) {
            assert(i < cols);
            assert(j < rows);
            char delim;
            ss >> m.data[i][j] >> delim;
        }
    }
    return m;
}

matrix matrix_init(size_t rows, size_t cols) {
    double* core = (double*) calloc(rows * cols, sizeof(double));
    double** data = (double**) malloc(rows * sizeof(double*));
    for (size_t i = 0; i < rows; i++) {
        data[i] = &core[i*cols];
    }
    matrix m = {
        core,
        data,
        rows,
        cols
    };
    return m;
}

void set_row(matrix m, size_t row, double* data) {
    for (size_t i = 0; i < m.cols; i++) {
        m.data[row][i] = data[i];
    }
}

void set_col(matrix m, size_t col, double* data) {
    for (size_t i = 0; i < m.cols; i++) {
        m.data[i][col] = data[i];
    }
}

matrix matmul(matrix m1, matrix m2) {

    assert(m1.cols == m2.rows);
    matrix res = matrix_init(m1.rows, m2.cols);

    for (size_t i = 0; i < res.rows; i++) {
        for (size_t j = 0; j < res.cols; j++) {
            for (size_t k = 0; k < m2.rows; k++) {
                res.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
    }

    return res;
}

matrix matadd(matrix m1, matrix m2) {
    assert(m1.rows == m2.rows);
    assert(m1.cols == m2.cols);
    matrix res = matrix_init(m1.rows, m1.cols);

    for (size_t i = 0; i < res.rows; i++) {
        for (size_t j = 0; j < res.cols; j++) {
            res.data[i][j] = m1.data[i][j] + m2.data[i][j];
        }
    }

    return res;
}

void matmul_inplace(matrix m1, matrix m2) {

    assert(m1.cols == m2.rows);
    matrix res = matmul(m1, m2);
    free(m1.core);
    free(m1.data);
    m1.core = res.core;
    m1.data = res.data;
}

void matadd_inplace(matrix m1, matrix m2) {
    assert(m1.rows == m2.rows);
    assert(m1.cols == m2.cols);

    for (size_t i = 0; i < m1.rows; i++) {
        for (size_t j = 0; j < m1.cols; j++) {
            m1.data[i][j] += m2.data[i][j];
        }
    }
}
