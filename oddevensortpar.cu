
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

constexpr unsigned int arr_size = 100000; // Number of elements in the input


__global__ void OddEvenSortSingleBlockKernel(int *numbers, int size)
{
	int stride = blockDim.x;

	for (int phase = 0; phase < size; ++phase)
	{
		int i = threadIdx.x;
		for (; i < size; i += stride)
		{
			if ((phase & 1) == 0)
			{
				// Even phase
				if (i % 2 == 0 && i + 1 < size)
				{
					if (numbers[i] > numbers[i + 1])
					{
						// Swap
						int temp = numbers[i];
						numbers[i] = numbers[i + 1];
						numbers[i + 1] = temp;
					}
				}
			}
			else
			{
				// Odd phase
				if (i % 2 == 1 && i + 1 < size)
				{
					if (numbers[i] > numbers[i + 1])
					{
						// Swap
						int temp = numbers[i];
						numbers[i] = numbers[i + 1];
						numbers[i + 1] = temp;
					}
				}
			}
		}
		
		__syncthreads();
	}
}

__global__ void OddEvenSortMultiBlockKernel(int *numbers, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= size)
		return;

	for (int phase = 0; phase < size; ++phase)
	{
		if ((phase & 1) == 0)
		{
			// Even phase
			if (i % 2 == 0 && i + 1 < size)
			{
				if (numbers[i] > numbers[i + 1])
				{
					// Swap
					int temp = numbers[i];
					numbers[i] = numbers[i + 1];
					numbers[i + 1] = temp;
				}
			}
		}
		else
		{
			// Odd phase
			if (i % 2 == 1 && i + 1 < size)
			{
				if (numbers[i] > numbers[i + 1])
				{
					// Swap
					int temp = numbers[i];
					numbers[i] = numbers[i + 1];
					numbers[i + 1] = temp;
				}
			}
		}
		
		__syncthreads();
	}
}


cudaError_t OddEvenSortCuda(std::vector<int> &numbers)
{
	int size = static_cast<int>(numbers.size());

	int *dev_numbersIn = nullptr;

	cudaError_t ret = cudaMalloc((void **)&dev_numbersIn, size * sizeof(int));
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		return ret;
	}

	ret = cudaMemcpy(dev_numbersIn, numbers.data(), size * sizeof(int), cudaMemcpyHostToDevice);
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		cudaFree(dev_numbersIn);
		return ret;
	}

	OddEvenSortSingleBlockKernel<<<1, 1024>>>(dev_numbersIn, size);

	/*
	constexpr int threadsPerBlock = 1024;
	constexpr int blocks = (arr_size + threadsPerBlock - 1) / threadsPerBlock;
	OddEvenSortSingleBlockKernel<<<blocks, threadsPerBlock>>>(dev_numbersIn, size);
	*/

	ret = cudaGetLastError();
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		cudaFree(dev_numbersIn);
		return ret;
	}

	ret = cudaDeviceSynchronize();
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		cudaFree(dev_numbersIn);
		return ret;
	}

	ret = cudaMemcpy(numbers.data(), dev_numbersIn, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		cudaFree(dev_numbersIn);
		return ret;
	}

	cudaFree(dev_numbersIn);
	return cudaSuccess;
}

void print_sort_status(std::vector<int> &numbers)
{ 
	std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

int main()
{
	cudaError_t ret;

	ret = cudaSetDevice(0);
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		return 1;
	}

	std::vector<int> numbers(arr_size);

	srand(static_cast<unsigned int>(time(0)));
	std::generate(numbers.begin(), numbers.end(), rand);

	print_sort_status(numbers);

	auto start = std::chrono::steady_clock::now();
	ret = OddEvenSortCuda(numbers);
	auto end = std::chrono::steady_clock::now();

	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		return 1;
	}

	print_sort_status(numbers);

	std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";

	ret = cudaDeviceReset();
	if (ret != cudaSuccess)
	{
		std::cout << "CUDA error (" << __LINE__ << "): " << cudaGetErrorString(ret) << std::endl;
		return 1;
	}

	return 0;
}
