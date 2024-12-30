/*
 This is all old code from when I first had the idea for this project. I didn't get very far because of how daunting the project seemed. I have since rethought my approach, but I might reuse some of the code from before, so I'm leaving it here.
*/

/*
Reference for commands to compile and run executable from root directory:
	nvcc gpu/main.cu -o gpu/main; echo "----Output----"; gpu/main.exe
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define NRA 32 // number of rows in matrix A
#define NCA 64 // number of columns in matrix A = number of rows in matrix B
#define NCB 32 // number of columns in matrix B
#define TILE_WIDTH 16	// dimension of shared memory array

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

void allocateMatrix(Matrix* matrix, int rows, int cols) {
	matrix->width = cols;
	matrix->height = rows;
	matrix->stride = cols;
	cudaMallocManaged(&matrix->elements, rows * cols * sizeof(float));
}

void initializeMatrixCPU(Matrix* matrix)
{
	int width = matrix->width;
	int height = matrix->height;
	int stride = matrix->stride;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			matrix->elements[i * stride + j] = i+j;
}

void printMatrix(Matrix* matrix) {
	for (int i = 0; i < matrix->height; i++) {
		printf("[");
		for (int j = 0; j < matrix->width; j++) {
			printf("%.2f ", matrix->elements[i * matrix->stride + j]);
		}
		printf("]\n");
	}
}

__device__ float getElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.stride + col];
}

__device__ void setElement(Matrix A, int row, int col, float value)
{
	A.elements[row * A.stride + col] = value;
}

// Returns the submatrix (tile) of size TILE_WIDTH*TILE_WIDTH located col tiles to the right and row tiles down.
__device__ Matrix getSubMatrix(const Matrix A, int row, int col)
{
	Matrix Asub;
	Asub.width = TILE_WIDTH;
	Asub.height = TILE_WIDTH;
	Asub.stride = A.stride;
	Asub.elements = &A.elements[row * TILE_WIDTH * A.stride + // Increments of TILE_WIDTH * A.stride
								col * TILE_WIDTH];			  // Increments of TILE_WIDTH
	return Asub;
}

/* 
*  Each thread computes one element of C = AB.
*  This kernel uses tiling to reduce number of reads from A and B, since they are in global memory. 
*  Tiling is the use of shared memory arrays of size TILE_WIDTH*TILE_WIDTH. 
*  Each thread reads one element from A and B into tiles As and Bs. Each thread block performs computation only using values in these two tiles.
*  Assumption: A.width (num columns of A) = B.height (num rows of B), and they are a multiple of TILE_WIDTH
*/
__global__ void matrixMultiply(Matrix C, Matrix A, Matrix B)
{
	// Each thread block computes a submatrix C_sub of C
	// C_sub.elements points to corresponding part of C.elements
	int tileRow = blockIdx.y; int tileCol = blockIdx.x;
	Matrix C_sub = getSubMatrix(C, tileRow, tileCol);

	// Each thread computes one element of C_sub
	int row = threadIdx.y; int col = threadIdx.x;
	float value = 0;

	printf("Hello World\n");

	// Compute the sum across every tile
	int numTiles = A.width / TILE_WIDTH;
	for (int tileNum = 0; tileNum < numTiles; tileNum++)
	{
		Matrix A_sub = getSubMatrix(A, tileRow, tileNum);	// A has numTiles tiles in each row
		Matrix B_sub = getSubMatrix(B, tileNum, tileCol);	// B has numTiles tiles in each column

		// Shared 2D array for elements of A_sub and B_sub
		__shared__ float As[TILE_WIDTH][TILE_WIDTH];
		__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

		// Each thread reads one element from each submatrix i.e. two total global memory access
		As[row][col] = getElement(A_sub, row, col);
		Bs[row][col] = getElement(B_sub, row, col);

		// Synchronize to ensure As and Bs are loaded correctly within each block.
		__syncthreads();

		// Compute product from corresponding elements of As and Bs, adding to existing value
		for (int k = 0; k < TILE_WIDTH; k++)
			value += As[row][k] * Bs[k][col];
		
		// Synchronize to ensure As and Bs are not overwritten before previous computation completes.
		__syncthreads();
	}
	setElement(C_sub, row, col, value);
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

void matrixMultiplicationTest()
{

	Matrix A, B, C;
	allocateMatrix(&A, NRA, NCA); allocateMatrix(&B, NCA, NCB); allocateMatrix(&C, NRA, NCB);
	initializeMatrixCPU(&A); initializeMatrixCPU(&B); initializeMatrixCPU(&C);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(B.width / TILE_WIDTH, A.height / TILE_WIDTH);
	matrixMultiply << <dimGrid, dimBlock >> > (C, A, B);

	printMatrix(&A); printMatrix(&B);
	printf("\n");
	printMatrix(&C);
	cudaFree(A.elements); cudaFree(B.elements); cudaFree(C.elements);
}

int main() {
	 vectorMultiplicationTest();
	//matrixMultiplicationTest();
	return 0;
}