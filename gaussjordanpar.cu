
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <algorithm>
//#include <iostream>
#include <chrono>
#include <stdio.h>


#define MAX_SIZE 4096

typedef double matrix[MAX_SIZE * MAX_SIZE];

int N;              /* matrix size			*/
int maxnum;         /* max number of element*/
char *Init;         /* matrix init type		*/
int PRINT;          /* print switch			*/
matrix A;           /* matrix A				*/
double b[MAX_SIZE]; /* vector b             */
double y[MAX_SIZE]; /* vector y             */


__host__ __device__ int CoordToIndex(int row, int col, int n)
{
	return row * n + col;
}


__global__ void GaussJordanDivKernel(double *mat, double *b, double *y, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int j = (k + 1) + idx;

    if (j >= n)
        return;

    int pivotIndex = CoordToIndex(k, k, n);
    double invPivotValue = 1.0 / mat[pivotIndex];

    mat[CoordToIndex(k, j, n)] *= invPivotValue;
}

__global__ void GaussJordanIntermediateKernel(double *mat, double *b, double *y, int n, int k)
{
    int pivotIndex = CoordToIndex(k, k, n);
    y[k] = b[k] / mat[pivotIndex];
    mat[pivotIndex] = 1.0;
}

__global__ void GaussJordanElimKernel(double *mat, double *b, double *y, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx;

    if (i >= n)
        return;

    if (i == k)
        return;

    int rowIndex = CoordToIndex(i, k, n);
	double pivotValue = mat[rowIndex];

    for (int j = k + 1; j < n; j++)
    {
        mat[CoordToIndex(i, j, n)] -= pivotValue * mat[CoordToIndex(k, j, n)];
    }

    if (i > k)
    {
        b[i] -= pivotValue * y[k];
    }
    else if (i < k)
    {
        y[i] -= pivotValue * y[k]; // HACK
    }
    
    mat[rowIndex] = 0.0;
}


cudaError_t work()
{
    double *dev_matrix = nullptr;
	double *dev_b = nullptr;
	double *dev_y = nullptr;

    size_t size = static_cast<size_t>(N) * N;

    cudaError_t ret = cudaMalloc((void **)&dev_matrix, sizeof(double) * size);
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        return ret;
    }
    
    ret = cudaMalloc((void **)&dev_b, sizeof(double) * N);
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        cudaFree(dev_matrix);
        return ret;
    }
    
    ret = cudaMalloc((void **)&dev_y, sizeof(double) * N);
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        return ret;
    }

    ret = cudaMemcpy(dev_matrix, A, sizeof(double) * size, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
    }

	ret = cudaMemcpy(dev_b, b, sizeof(double) * N, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
	}


    for (int k = 0; k < N; k++)
    {
        const unsigned int threadCount = 128;
        const unsigned int blockCount = (N + threadCount - 1) / threadCount;

        GaussJordanDivKernel<<<blockCount, threadCount>>>(dev_matrix, dev_b, dev_y, N, k);

        ret = cudaGetLastError();
        if (ret != cudaSuccess)
        {
            printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
            cudaFree(dev_matrix);
            cudaFree(dev_b);
            cudaFree(dev_y);
            return ret;
        }


        GaussJordanIntermediateKernel<<<1, 1>>>(dev_matrix, dev_b, dev_y, N, k);

        ret = cudaGetLastError();
        if (ret != cudaSuccess)
        {
            printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
            cudaFree(dev_matrix);
            cudaFree(dev_b);
            cudaFree(dev_y);
            return ret;
        }


        GaussJordanElimKernel<<<blockCount, threadCount>>>(dev_matrix, dev_b, dev_y, N, k);

        ret = cudaGetLastError();
        if (ret != cudaSuccess)
        {
            printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
            cudaFree(dev_matrix);
            cudaFree(dev_b);
            cudaFree(dev_y);
            return ret;
        }
    }
    

    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
    }

    cudaFree(dev_b);

    ret = cudaMemcpy(A, dev_matrix, sizeof(double) * size, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        cudaFree(dev_matrix);
        cudaFree(dev_y);
        return ret;
    }

	ret = cudaMemcpy(y, dev_y, sizeof(double) * N, cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
    {
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
        cudaFree(dev_matrix);
        cudaFree(dev_y);
        return ret;
	}

    cudaFree(dev_matrix);
    cudaFree(dev_y);
    return cudaSuccess;
}


void Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < std::min(N, 16); i++)
    {
        printf("[");
        for (j = 0; j < std::min(N, 16); j++)
            printf(" %5.2f,", A[CoordToIndex(i, j, N)]);
        printf("]\n");
    }

    printf("Vector y:\n[");
    for (j = 0; j < std::min(N, 16); j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");

    printf("\n\n");
}

void Init_Matrix()
{
    int i, j;

    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init, "rand") == 0)
    {
        srand(0);

        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /* diagonal dominance */
                {
                    A[CoordToIndex(i, j, N)] = (double)(rand() % maxnum) + 5.0;
                }
                else
                {
                    A[CoordToIndex(i, j, N)] = (double)(rand() % maxnum) + 1.0;
                }
            }
        }

        srand(time(0));
    }

    if (strcmp(Init, "fast") == 0)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /* diagonal dominance */
                {
                    A[CoordToIndex(i, j, N)] = 5.0;
                }
                else
                {
                    A[CoordToIndex(i, j, N)] = 2.0;
                }
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++)
    {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    printf("done \n\n");

    if (PRINT == 1)
        Print_Matrix();
}

void Init_Default()
{
    N = 2048;
    //Init = "fast";
    Init = "rand";
    maxnum = 15.0;
    PRINT = 1;
}

int Read_Options(int argc, char **argv)
{
    char *prog;

    prog = *argv;
    while (++argv, --argc > 0)
    {
        if (**argv == '-')
        {
            switch (*++ * argv)
            {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: gaussian [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand");
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
                exit(0);
                break;
            case 'I':
                --argc;
                Init = *++argv;
                break;
            case 'm':
                --argc;
                maxnum = atoi(*++argv);
                break;
            case 'P':
                --argc;
                PRINT = atoi(*++argv);
                break;
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }
        }
    }
	return 0;
}

int main(int argc, char **argv)
{
    printf("Gauss Jordan Par\n");

    Init_Default();           /* Init default values	*/
    Read_Options(argc, argv); /* Read arguments	*/
    Init_Matrix();            /* Init the matrix	*/

	cudaError_t ret = cudaSetDevice(0);
	if (ret != cudaSuccess)
	{
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
		return 1;
	}

	auto start = std::chrono::steady_clock::now();
	ret = work();
	auto end = std::chrono::steady_clock::now();

	if (ret != cudaSuccess)
	{
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
		return 1;
	}

    if (PRINT == 1)
        Print_Matrix();

    printf("Elapsed time = %f sec\n", std::chrono::duration<double>(end - start).count());

	ret = cudaDeviceReset();
	if (ret != cudaSuccess)
	{
        printf("CUDA error (%d): %s\n", __LINE__, cudaGetErrorString(ret));
		return 1;
	}

	return 0;
}
