typedef struct {
  double** data;
  int rows;
  int cols;
} matrix;

/* Initialise a matrix. */
matrix matrix_init(unsigned rows, unsigned cols);

/* Set a row of matrix m. */
void set_row(matrix m, int row, double* data);

/* Set a column of matrix m. */
void set_col(matrix m, int col, double* data);

/* Get the result of a matrix multiplication of matrices m1 and m2. */
matrix matmul(matrix m1, matrix m2);
