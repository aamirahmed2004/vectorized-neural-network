#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define NRA 20 // number of rows in matrix A
#define NCA 30 // number of columns in matrix A = number of rows in matrix B
#define NCB 40 // number of columns in matrix B
#define BLOCK_SIZE 16	// dimension of shared memory array

#define SCALAR_MULTIPLY 1
#define ELEMENTWISE_MULTIPLY 2

/*
*  Matrix elements accessed using row-major order: 
*		M[row][col] = M.elements[row * M.stride + col] = *(M.elements + (row * M.stride + col))
*  In case the instantiated matrix is a submatrix, stride represents the width of the original matrix, for consistent memory access. 
*  If the instantiated matrix is the original matrix, stride = width.
*/
typedef struct {
	int width;
	int height;
	int stride;		
	float* elements;
} Matrix;

__device__ float getElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

__device__ void setElement(Matrix A, int row, int col, float value)
{
	A.elements[row * A.stride + col] = value;
}

// Returns the submatrix (tile) of size BLOCK_SIZE*BLOCK_SIZE col tiles to the right and row tiles down.
// Assumes stride is a multiple of BLOCK_SIZE
__device__ Matrix getSubMatrix(const Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[row * BLOCK_SIZE * A.stride + // Increments of BLOCK_SIZE * A.stride
								col * BLOCK_SIZE];			  // Increments of BLOCK_SIZE
	return Asub;
}

// Forward declaration of device functions
__device__ void elementwiseMultiply(float* c, const float* a, const float* b, const int numElems);
__device__ void scalarMultiply(float* c, const float a, const float* b, const int numElems);

__global__ void multiply(float* c, const float* a, const float* b, const float scalar, const int numElems, const int mode) 
{
	if (mode == SCALAR_MULTIPLY) 
		scalarMultiply(c, scalar, b, numElems);

	else if (mode == ELEMENTWISE_MULTIPLY)
		elementwiseMultiply(c, a, b, numElems);
}

__device__ void elementwiseMultiply(float* c, const float* a, const float* b, const int numElems) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numElems) {
		c[i] = a[i] * b[i];
	}
}

__device__ void scalarMultiply(float* c, const float a, const float* b, const int numElems) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numElems) {
		c[i] = a * b[i];
	}
}

void vectorMultiplicationTest() {
	int N = 1 << 20;
	float scalar = 2.0f;
	size_t vectorBytes = N * sizeof(float);

	// Allocate memory on GPU 
	float* a; float* b; float* c;
	cudaMallocManaged(&a, vectorBytes); cudaMallocManaged(&b, vectorBytes); cudaMallocManaged(&c, vectorBytes);

	// Initialize vectors on the CPU
	for (int i = 0; i < N; i++) {
		a[i] = 2.0f;
		b[i] = 3.0f;
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	multiply << < blocksPerGrid, threadsPerBlock >> > (c, a, b, scalar, N, SCALAR_MULTIPLY);
	cudaDeviceSynchronize();

	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = std::max(maxError, abs(c[i] - 6.0f));
	}

	printf("Max error: %.2f\n", maxError);
	printf("First and last element: %.2f  |  %.2f\n", c[0], c[N - 1]);

	cudaFree(a); cudaFree(b); cudaFree(c);
}


void initializeMatrix(int rows, int cols, float** matrix, int value) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			matrix[i][j] = value;
		}
	}
}

int main() {
	 //vectorMultiplicationTest();
	
	return 0;
}