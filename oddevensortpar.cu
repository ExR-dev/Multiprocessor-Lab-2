
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>

constexpr unsigned int arr_size = 524288; // Number of elements in the input
constexpr unsigned int thread_count = (1 << 10);
constexpr unsigned int block_count = ((arr_size / 2) + thread_count - 1) / thread_count;

#define MULTI_BLOCK_SORT 1


__global__ void OddEvenSortSingleBlockKernel(int *numbers, int size)
{
	int stride = blockDim.x * 2;

	for (int phase = 0; phase < size; ++phase)
	{
		int phaseOffset = phase & 1; // 0 for even phase, 1 for odd phase

		for (int i = threadIdx.x * 2 + phaseOffset; i < size; i += stride)
		{
			if (i + 1 >= size)
				continue;
			
			if (numbers[i] > numbers[i + 1])
			{
				// Swap
				int temp = numbers[i];
				numbers[i] = numbers[i + 1];
				numbers[i + 1] = temp;
			}
		}
		
		__syncthreads();
	}
}

__global__ void OddEvenSortMultiBlockKernel(int *numbers, int size, int iter)
{
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + (iter & 1);

	if (i + 1 >= size)
		return;
		
	if (numbers[i] > numbers[i + 1])
	{
		// Swap
		int temp = numbers[i];
		numbers[i] = numbers[i + 1];
		numbers[i + 1] = temp;
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

	if (MULTI_BLOCK_SORT)
	{
		for (int iter = 0; iter < size; ++iter)
		{
			OddEvenSortMultiBlockKernel<<<block_count, thread_count>>>(dev_numbersIn, size, iter);
		}
	}
	else
	{
		OddEvenSortSingleBlockKernel<<<1, thread_count>>>(dev_numbersIn, size);
	}

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
