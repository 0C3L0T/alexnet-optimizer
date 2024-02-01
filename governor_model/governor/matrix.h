#ifndef EMBEDDEDSYSTEMS_MATRIX_H
#define EMBEDDEDSYSTEMS_MATRIX_H

#include <stdlib.h>
#include <string>

typedef struct {
  double* core;
  double** data;
  size_t rows;
  size_t cols;
} matrix;

/* Clean up the memory of a matrix. */
void matrix_destroy(matrix m);

/* Build a matrix from a file. */
matrix matrix_from_file(std::string filename, size_t rows, size_t cols);

/* Initialise a matrix. */
matrix matrix_init(size_t rows, size_t cols);


/* Set a row of matrix m. */
void set_row(matrix m, size_t row, double* data);

/* Set a column of matrix m. */
void set_col(matrix m, size_t col, double* data);

/* Get the result of a matrix multiplication of matrices m1 and m2. */
matrix matmul(matrix m1, matrix m2);

/* Get the result of the addition of matrices m1 and m2. */
matrix matadd(matrix m1, matrix m2);

/* Matrix multiply m1 and m2, storing the result in m1. */
void matmul_inplace(matrix m1, matrix m2);

/* Add m1 and m2, storing the result in m1. */
void matadd_inplace(matrix m1, matrix m2);

#endif