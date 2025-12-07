
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <stdio.h>


#define MAX_SIZE 4096

typedef double matrix[MAX_SIZE][MAX_SIZE];

int N;              /* matrix size			*/
int maxnum;         /* max number of element*/
char *Init;         /* matrix init type		*/
int PRINT;          /* print switch			*/
matrix A;           /* matrix A				*/
double b[MAX_SIZE]; /* vector b             */
double y[MAX_SIZE]; /* vector y             */


__device__ int CoordToIndex(int row, int col, int n)
{
    return row * n + col;
}

__device__ int IndexToRow(int index, int n)
{
	return index / n;
}

__device__ int IndexToCol(int index, int n)
{
    return index % n;
}


__global__ void GaussianJordanKernel(double **mat, double *b, double *y, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int k = 0; k < N; k++) /* Outer loop */
    {
        for (int j = k + 1; j < N; j++) /* Division step */
        {
            mat[k][j] = mat[k][j] / mat[k][k];
        }

        y[k] = b[k] / mat[k][k];
        mat[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++) /* Elimination step */
            {
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            }

            b[i] = b[i] - mat[i][k] * y[k];
            mat[i][k] = 0.0;
        }

        for (int i = 0; i < k; i++)
        {
            for (int j = k + 1; j < N; j++) /* Additional Elimination for Gauss-Jordan */
            {
                mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
            }

            y[i] = y[i] - mat[i][k] * y[k];
            mat[i][k] = 0.0;
        }
    }
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
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        return ret;
    }
    
    ret = cudaMalloc((void **)&dev_b, sizeof(double) * MAX_SIZE);
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        return ret;
    }
    
    ret = cudaMalloc((void **)&dev_y, sizeof(double) * MAX_SIZE);
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        return ret;
    }


    ret = cudaMemcpy(dev_matrix, A, sizeof(double) * size, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
    }

	ret = cudaMemcpy(dev_b, b, sizeof(double) * MAX_SIZE, cudaMemcpyHostToDevice);
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
	}

	const unsigned int threadCount = 128;
    const unsigned int blockCount = (N + threadCount - 1) / threadCount;

    GaussianJordanKernel<<<blockCount, threadCount>>>(dev_matrix, dev_b, dev_y, N);

    ret = cudaGetLastError();
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
    }


    ret = cudaDeviceSynchronize();
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
    }


    ret = cudaMemcpy(A, dev_matrix, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
    }

	ret = cudaMemcpy(y, dev_y, MAX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
    {
        std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
        cudaFree(dev_matrix);
        cudaFree(dev_b);
        cudaFree(dev_y);
        return ret;
	}


    cudaFree(dev_matrix);
    cudaFree(dev_b);
    cudaFree(dev_y);
    return cudaSuccess;
}

void Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++)
    {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i][j]);
        printf("]\n");
    }

    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
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
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /* diagonal dominance */
                {
                    A[i][j] = (double)(rand() % maxnum) + 5.0;
                }
                else
                {
                    A[i][j] = (double)(rand() % maxnum) + 1.0;
                }
            }
        }
    }

    if (strcmp(Init, "fast") == 0)
    {
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                if (i == j) /* diagonal dominance */
                {
                    A[i][j] = 5.0;
                }
                else
                {
                    A[i][j] = 2.0;
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
    Init = "fast";
    maxnum = 15.0;
    PRINT = 0;
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
}

int main(int argc, char **argv)
{
    printf("Gauss Jordan\n");

    Init_Default();           /* Init default values	*/
    Read_Options(argc, argv); /* Read arguments	*/
    Init_Matrix();            /* Init the matrix	*/

	cudaError_t ret = cudaSetDevice(0);
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		return 1;
	}

	auto start = std::chrono::steady_clock::now();
	ret = work();
	auto end = std::chrono::steady_clock::now();

	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		return 1;
	}

    if (PRINT == 1)
        Print_Matrix();

	std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";

	ret = cudaDeviceReset();
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		return 1;
	}

	return 0;
}
