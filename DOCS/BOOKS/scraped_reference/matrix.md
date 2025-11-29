---
title: "Matrices and vectors"
url: "https://www.mql5.com/en/docs/matrix"
hierarchy: []
scraped_at: "2025-11-28 09:30:54"
---

# Matrices and vectors

[MQL5 Reference](/en/docs "MQL5 Reference")Matrix and Vector Methods

* [Matrix and Vector Types](/en/docs/matrix/matrix_types "Matrix and Vector Types")
* [Initialization](/en/docs/matrix/matrix_initialization "Initialization")
* [Manipulations](/en/docs/matrix/matrix_manipulations "Manipulations")
* [Operations](/en/docs/matrix/matrix_operations "Operations")
* [Products](/en/docs/matrix/matrix_products "Products")
* [Transformations](/en/docs/matrix/matrix_decompositions "Transformations")
* [Statistics](/en/docs/matrix/matrix_statistics "Statistics")
* [Features](/en/docs/matrix/matrix_characteristics "Features")
* [Matrix Classification](/en/docs/matrix/matrix_classification "Matrix Classification")
* [Solutions](/en/docs/matrix/matrix_solves "Solutions")
* [Machine learning](/en/docs/matrix/matrix_machine_learning "Machine learning")
* [OpenBLAS](/en/docs/matrix/openblas "OpenBLAS")

# Matrices and vectors

A matrix is a two-dimensional array of double, float, or complex numbers.

A vector is a one-dimensional array of double, float, or complex numbers. The vector has no indication of whether it is vertical or horizontal. It is determined from the use context. For example, the vector operation Dot assumes that the left vector is horizontal and the right one is vertical. If the type indication is required, one-row or one-column matrices can be used. However, this is generally not necessary.

Matrices and vectors allocate memory for data dynamically. In fact, matrices and vectors are objects that have certain properties, such as the type of data they contain and dimensions. Matrix and vector properties can be obtained using methods such as vector\_a.Size(), matrix\_b.Rows(), vector\_c.Norm(), matrix\_d.Cond() and others. Any dimension can be changed.

When creating and initializing matrices, so-called static methods are used (these are like static methods of a class). For example: matrix::Eye(), matrix::Identity(), matrix::Ones(), vector::Ones(), matrix: :Zeros(), vector::Zeros(), matrix::Full(), vector::Full(), matrix::Tri().

At the moment, matrix and vector operations do not imply the use of the complex data type, as this development direction has not yet been completed.

MQL5 supports passing of matrices and vectors to DLLs. This enables the import of functions utilizing the relevant types, from external variables.

Matrices and vectors are passed to a DLL as a pointer to a buffer. For example, to pass a matrix of type float, the corresponding parameter of the function exported from the DLL must take a float-type buffer pointer.

MQL5

| |
| --- |
| #import "mmlib.dll" bool sgemm(uint flags, matrix<float> &C, const matrix<float> &A, const matrix<float> &B, ulong M, ulong N, ulong K, float alpha, float beta); #import |

C++

| |
| --- |
| extern "C" \_\_declspec(dllexport) bool sgemm(UINT flags, float \*C, const float \*A, const float \*B, UINT64 M, UINT64 N, UINT64 K, float alpha, float beta) |

In addition to buffers, you should pass matrix and vector sizes for correct processing.

All matrix and vector methods are listed below in alphabetical order.

| Function | Action | Category |
| --- | --- | --- |
| [Activation](/en/docs/matrix/matrix_machine_learning/matrix_activation) | Compute activation function values and write them to the passed vector/matrix | [Machine learning](/en/docs/matrix/matrix_machine_learning) |
| [ArgMax](/en/docs/matrix/matrix_statistics/matrix_argmax) | Return the index of the maximum value | [Statistics](/en/docs/matrix/matrix_statistics) |
| [ArgMin](/en/docs/matrix/matrix_statistics/matrix_argmin) | Return the index of the minimum value | [Statistics](/en/docs/matrix/matrix_statistics) |
| [ArgSort](/en/docs/matrix/matrix_manipulations/matrix_argsort) | Return the sorted index | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Assign](/en/docs/matrix/matrix_initialization/matrix_assign) | Copies a matrix, vector or array with auto cast | [Initialization](/en/docs/matrix/matrix_initialization) |
| [Average](/en/docs/matrix/matrix_statistics/matrix_average) | Compute the weighted average of matrix/vector values | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Cholesky](/en/docs/matrix/matrix_decompositions/matrix_cholesky) | Compute the Cholesky decomposition | [Transformations](/en/docs/matrix/matrix_decompositions) |
| [Clip](/en/docs/matrix/matrix_manipulations/matrix_clip) | Limits the elements of a matrix/vector to a given range of valid values | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Col](/en/docs/matrix/matrix_manipulations/matrix_col) | Return a column vector. Write a vector to the specified column. | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Cols](/en/docs/matrix/matrix_characteristics/matrix_cols) | Return the number of columns in a matrix | [Features](/en/docs/matrix/matrix_characteristics) |
| [Compare](/en/docs/matrix/matrix_manipulations/matrix_compare) | Compare the elements of two matrices/vectors with the specified precision | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [CompareByDigits](/en/docs/matrix/matrix_manipulations/matrix_comparebydigits) | Compare the elements of two matrices/vectors with the significant figures precision | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Cond](/en/docs/matrix/matrix_characteristics/matrix_cond) | Compute the condition number of a matrix | [Features](/en/docs/matrix/matrix_characteristics) |
| [Convolve](/en/docs/matrix/matrix_products/matrix_convolve) | Return the discrete, linear convolution of two vectors | [Products](/en/docs/matrix/matrix_products) |
| [Copy](/en/docs/matrix/matrix_manipulations/matrix_copy) | Return a copy of the given matrix/vector | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Concat](/en/docs/matrix/matrix_manipulations/matrix_concat) | Concatenate 2 submatrices to one matrix. Concatenate 2 vectors to one vector | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [CopyIndicatorBuffer](/en/docs/matrix/matrix_initialization/matrix_copyindicatorbuffer) | Get the data of the specified [indicator](/en/docs/indicators) buffer in the specified quantity to a [vector](/en/docs/basis/types/matrix_vector) | [Initialization](/en/docs/matrix/matrix_initialization) |
| [CopyRates](/en/docs/matrix/matrix_initialization/matrix_copyrates) | Gets the historical series of the [MqlRates](/en/docs/constants/structures/mqlrates) structure of the specified symbol-period in the specified amount into a matrix or vector | [Initialization](/en/docs/matrix/matrix_initialization) |
| [CopyTicks](/en/docs/matrix/matrix_initialization/matrix_copyticks) | Get ticks from an [MqlTick](/en/docs/constants/structures/mqltick) structure into a matrix or a vector | [Initialization](/en/docs/matrix/matrix_initialization) |
| [CopyTicksRange](/en/docs/matrix/matrix_initialization/matrix_copyticksrange) | Get ticks from an [MqlTick](/en/docs/constants/structures/mqltick) structure into a matrix or a vector within the specified date range | [Initialization](/en/docs/matrix/matrix_initialization) |
| [CorrCoef](/en/docs/matrix/matrix_products/matrix_corrcoef) | Compute the Pearson correlation coefficient (linear correlation coefficient) | [Products](/en/docs/matrix/matrix_products) |
| [Correlate](/en/docs/matrix/matrix_products/matrix_correlate) | Compute the cross-correlation of two vectors | [Products](/en/docs/matrix/matrix_products) |
| [Cov](/en/docs/matrix/matrix_products/matrix_cov) | Compute the covariance matrix | [Products](/en/docs/matrix/matrix_products) |
| [CumProd](/en/docs/matrix/matrix_statistics/matrix_cumprod) | Return the cumulative product of matrix/vector elements, including those along the given axis | [Statistics](/en/docs/matrix/matrix_statistics) |
| [CumSum](/en/docs/matrix/matrix_statistics/matrix_cumsum) | Return the cumulative sum of matrix/vector elements, including those along the given axis | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Derivative](/en/docs/matrix/matrix_machine_learning/matrix_derivative) | Compute activation function derivative values and write them to the passed vector/matrix | [Machine learning](/en/docs/matrix/matrix_machine_learning) |
| [Det](/en/docs/matrix/matrix_characteristics/matrix_det) | Compute the determinant of a square invertible matrix | [Features](/en/docs/matrix/matrix_characteristics) |
| [Diag](/en/docs/matrix/matrix_manipulations/matrix_diag) | Extract a diagonal or construct a diagonal matrix | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Dot](/en/docs/matrix/matrix_products/matrix_dot) | Dot product of two vectors | [Products](/en/docs/matrix/matrix_products) |
| [Eig](/en/docs/matrix/matrix_decompositions/matrix_eig) | Computes the eigenvalues and right eigenvectors of a square matrix | [Transformations](/en/docs/matrix/matrix_decompositions) |
| [EigVals](/en/docs/matrix/matrix_decompositions/matrix_eigvals) | Computes the eigenvalues of a general matrix | [Transformations](/en/docs/matrix/matrix_decompositions) |
| [Eye](/en/docs/matrix/matrix_initialization/matrix_eye) | Return a matrix with ones on the diagonal and zeros elsewhere | [Initialization](/en/docs/matrix/matrix_initialization) |
| [Fill](/en/docs/matrix/matrix_initialization/matrix_fill) | Fill an existing matrix or vector with the specified value | [Initialization](/en/docs/matrix/matrix_initialization) |
| [Flat](/en/docs/matrix/matrix_manipulations/matrix_flat) | Access a matrix element through one index instead of two | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Full](/en/docs/matrix/matrix_initialization/matrix_full) | Create and return a new matrix filled with the given value | [Initialization](/en/docs/matrix/matrix_initialization) |
| [GeMM](/en/docs/matrix/matrix_products/matrix_gemm) | The GeMM (General Matrix Multiply) method implements the general multiplication of two matrices | [Products](/en/docs/matrix/matrix_products) |
| [HasNan](/en/docs/matrix/matrix_manipulations/matrix_hasnan) | Return the number of [NaN](/en/docs/basis/types/double) values in a matrix/vector | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Hsplit](/en/docs/matrix/matrix_manipulations/matrix_hsplit) | Split a matrix horizontally into multiple submatrices. Same as Split with axis=0 | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Identity](/en/docs/matrix/matrix_initialization/matrix_identity) | Create an identity matrix of the specified size | [Initialization](/en/docs/matrix/matrix_initialization) |
| [Init](/en/docs/matrix/matrix_initialization/matrix_init) | Matrix or vector initialization | [Initialization](/en/docs/matrix/matrix_initialization) |
| [Inner](/en/docs/matrix/matrix_products/matrix_inner) | Inner product of two matrices | [Products](/en/docs/matrix/matrix_products) |
| [Inv](/en/docs/matrix/matrix_solves/matrix_inv) | Compute the multiplicative inverse of a square invertible matrix by the Jordan-Gauss method | [Solutions](/en/docs/matrix/matrix_solves) |
| [Kron](/en/docs/matrix/matrix_products/matrix_kron) | Return Kronecker product of two matrices, matrix and vector, vector and matrix or two vectors | [Products](/en/docs/matrix/matrix_products) |
| [Loss](/en/docs/matrix/matrix_machine_learning/matrix_loss) | Compute loss function values and write them to the passed vector/matrix | [Machine learning](/en/docs/matrix/matrix_machine_learning) |
| [LstSq](/en/docs/matrix/matrix_solves/matrix_lstsq) | Return the least-squares solution of linear algebraic equations (for non-square or degenerate matrices) | [Solutions](/en/docs/matrix/matrix_solves) |
| [LU](/en/docs/matrix/matrix_decompositions/matrix_lu) | Implement an LU decomposition of a matrix: the product of a lower triangular matrix and an upper triangular matrix | [Transformations](/en/docs/matrix/matrix_decompositions) |
| [LUP](/en/docs/matrix/matrix_decompositions/matrix_lup) | Implement an LUP factorization with partial permutation, which refers to LU decomposition with row permutations only: PA=LU | [Transformations](/en/docs/matrix/matrix_decompositions) |
| [MatMul](/en/docs/matrix/matrix_products/matrix_matmul) | Matrix product of two matrices | [Products](/en/docs/matrix/matrix_products) |
| [Max](/en/docs/matrix/matrix_statistics/matrix_max) | Return the maximum value in a matrix/vector | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Mean](/en/docs/matrix/matrix_statistics/matrix_mean) | Compute the arithmetic mean of element values | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Median](/en/docs/matrix/matrix_statistics/matrix_median) | Compute the median of the matrix/vector elements | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Min](/en/docs/matrix/matrix_statistics/matrix_min) | Return the minimum value in a matrix/vector | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Norm](/en/docs/matrix/matrix_characteristics/matrix_norm) | Return matrix or vector norm | [Features](/en/docs/matrix/matrix_characteristics) |
| [Ones](/en/docs/matrix/matrix_initialization/matrix_ones) | Create and return a new matrix filled with ones | [Initialization](/en/docs/matrix/matrix_initialization) |
| [Outer](/en/docs/matrix/matrix_products/matrix_outer) | Compute the outer product of two matrices or two vectors | [Products](/en/docs/matrix/matrix_products) |
| [Percentile](/en/docs/matrix/matrix_statistics/matrix_percentile) | Return the specified percentile of values of matrix/vector elements or elements along the specified axis | [Statistics](/en/docs/matrix/matrix_statistics) |
| [PInv](/en/docs/matrix/matrix_solves/matrix_pinv) | Compute the pseudo-inverse of a matrix by the Moore-Penrose method | [Solutions](/en/docs/matrix/matrix_solves) |
| [Power](/en/docs/matrix/matrix_products/matrix_power) | Raise a square matrix to an integer power | [Products](/en/docs/matrix/matrix_products) |
| [Prod](/en/docs/matrix/matrix_statistics/matrix_prod) | Return the product of matrix/vector elements, which can also be executed for the given axis | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Ptp](/en/docs/matrix/matrix_statistics/matrix_ptp) | Return the range of values of a matrix/vector or of the given matrix axis | [Statistics](/en/docs/matrix/matrix_statistics) |
| [QR](/en/docs/matrix/matrix_decompositions/matrix_qr) | Compute the qr factorization of a matrix | [Transformations](/en/docs/matrix/matrix_decompositions) |
| [Quantile](/en/docs/matrix/matrix_statistics/matrix_quantile) | Return the specified quantile of values of matrix/vector elements or elements along the specified axis | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Random](/en/docs/matrix/matrix_initialization/matrix_random) | Static function. Create and return a new matrix or vector filled with random values. Random values are generated uniformly within the specified range | [Initialization](/en/docs/matrix/matrix_initialization) |
| [Rank](/en/docs/matrix/matrix_characteristics/matrix_rank) | Return matrix rank using the Gaussian method | [Features](/en/docs/matrix/matrix_characteristics) |
| [RegressionMetric](/en/docs/matrix/matrix_machine_learning/matrix_regressionmetrics) | Compute the regression metric as the deviation error from the regression line constructed on the specified data array | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Reshape](/en/docs/matrix/matrix_manipulations/matrix_reshape) | Change the shape of a matrix without changing its data | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Resize](/en/docs/matrix/matrix_manipulations/matrix_resize) | Return a new matrix with a changed shape and size | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Row](/en/docs/matrix/matrix_manipulations/matrix_row) | Return a row vector. Write the vector to the specified row | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Rows](/en/docs/matrix/matrix_characteristics/matrix_rows) | Return the number of rows in a matrix | [Features](/en/docs/matrix/matrix_characteristics) |
| [Set](/en/docs/matrix/matrix_manipulations/matrix_set) | Sets the value for a vector element by the specified index | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Size](/en/docs/matrix/matrix_characteristics/matrix_size) | Return the size of vector | [Features](/en/docs/matrix/matrix_characteristics) |
| [SLogDet](/en/docs/matrix/matrix_characteristics/matrix_slogdet) | Compute the sign and logarithm of the determinant of an matrix | [Features](/en/docs/matrix/matrix_characteristics) |
| [Solve](/en/docs/matrix/matrix_solves/matrix_solve) | Solve a linear matrix equation or a system of linear algebraic equations | [Solutions](/en/docs/matrix/matrix_solves) |
| [Sort](/en/docs/matrix/matrix_manipulations/matrix_sort) | Sort by place | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Spectrum](/en/docs/matrix/matrix_characteristics/matrix_spectrum) | Compute spectrum of a matrix as the set of its eigenvalues from the product AT\*A | [Features](/en/docs/matrix/matrix_characteristics) |
| [Split](/en/docs/matrix/matrix_manipulations/matrix_split) | Split a matrix into multiple submatrices | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Std](/en/docs/matrix/matrix_statistics/matrix_std) | Return the standard deviation of values of matrix/vector elements or elements along the specified axis | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Sum](/en/docs/matrix/matrix_statistics/matrix_sum) | Return the sum of matrix/vector elements, which can also be executed for the given axis (axes) | [Statistics](/en/docs/matrix/matrix_statistics) |
| [SVD](/en/docs/matrix/matrix_decompositions/matrix_svd) | Singular value decomposition | [Transformations](/en/docs/matrix/matrix_decompositions) |
| [SwapCols](/en/docs/matrix/matrix_manipulations/matrix_swapcols) | Swap columns in a matrix | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [SwapRows](/en/docs/matrix/matrix_manipulations/matrix_swaprows) | Swap rows in a matrix | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Trace](/en/docs/matrix/matrix_characteristics/matrix_trace) | Return the sum along diagonals of the matrix | [Features](/en/docs/matrix/matrix_characteristics) |
| [Transpose](/en/docs/matrix/matrix_manipulations/matrix_transpose) | Transpose (swap the axes) and return the modified matrix | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Tri](/en/docs/matrix/matrix_initialization/matrix_tri) | Construct a matrix with ones on a specified diagonal and below, and zeros elsewhere | [Initialization](/en/docs/matrix/matrix_initialization) |
| [TriL](/en/docs/matrix/matrix_manipulations/matrix_tril) | Return a copy of a matrix with elements above the k-th diagonal zeroed. Lower triangular matrix | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [TriU](/en/docs/matrix/matrix_manipulations/matrix_triu) | Return a copy of a matrix with the elements below the k-th diagonal zeroed. Upper triangular matrix | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Var](/en/docs/matrix/matrix_statistics/matrix_var) | Compute the variance of values of matrix/vector elements | [Statistics](/en/docs/matrix/matrix_statistics) |
| [Vsplit](/en/docs/matrix/matrix_manipulations/matrix_vsplit) | Split a matrix vertically into multiple submatrices. Same as Split with axis=1 | [Manipulations](/en/docs/matrix/matrix_manipulations) |
| [Zeros](/en/docs/matrix/matrix_initialization/matrix_zeros) | Create and return a new matrix filled with zeros | [Initialization](/en/docs/matrix/matrix_initialization) |

[ArrayFromFP8](/en/docs/array/arrayfromfp8 "ArrayFromFP8")

[Matrix and Vector Types](/en/docs/matrix/matrix_types "Matrix and Vector Types")