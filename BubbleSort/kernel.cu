#include "cuda_runtime.h"

#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <ctime>
#include "device_launch_parameters.h"

using namespace std;

inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		cout << "CUDA Runtime Error: " << cudaGetErrorString(result) << endl;
	}

	return result;
}

__global__ void kernel(float *a, int arraySize, int k)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	int N = 2 * index;

	float tmp;

	if (k%2 == 1) {
		if (N + 2 < arraySize) {
			if (a[N + 1] > a[N + 2]) {
				tmp = a[N + 1];
				a[N + 1] = a[N + 2];
				a[N + 2] = tmp;
			}
		}
	}
	else {
		if (a[N] > a[N + 1]) {
			tmp = a[2 * index];
			a[N] = a[N + 1];
			a[N + 1] = tmp;
		}
	}
}

__global__ void kernel2(float* dev_tab, unsigned int N)
{
	int ind, i, j, s;
	float a, b;
	ind = 2 * (threadIdx.x + blockDim.x*blockIdx.x);
	for (unsigned int k = 0; k < N-1; k++)
	{
		s = (k % 2);
		i = ind + s;
		j = ind + 1 + s;
		if (j<N)
		{
			a = dev_tab[i];
			b = dev_tab[j];
			if (b<a)
			{
				dev_tab[i] = b;
				dev_tab[j] = a;
			}
		}
		__syncthreads();
	}
}

__global__ void kernel3(float* dev_tab, unsigned int N, const int Nt)
{
	extern __shared__ float stab[];
	int ind, i, j, s;
	float a, b;
	ind = 2 * threadIdx.x;

	i = ind;
	j = ind + 1;
	a = dev_tab[i];
	b = dev_tab[j];

	if (b < a) {
		stab[i] = b;
		stab[j] = a;
	} else {
		stab[i] = a;
		stab[j] = b;
	}
	__syncthreads();

	for (unsigned int k = 1; k < N-2; k++)
	{
		s = (k % 2);
		i = ind + s;
		j = ind + 1 + s;
		if (j<N)
		{
			a = stab[i];
			b = stab[j];
			if (b<a)
			{
				stab[i] = b;
				stab[j] = a;
			}
		}
		__syncthreads();
	}

	i = ind;
	j = ind + 1;
	a = stab[i];
	b = stab[j];

	if (b<a) {
		dev_tab[i] = b;
		dev_tab[j] = a;
	} else {
		dev_tab[i] = a;
		dev_tab[j] = b;
	}
}

void bubbleSortCpu(float* b, int n)
{
	int i, j;

	double tt;
	LARGE_INTEGER tb, te, tf;
	QueryPerformanceFrequency(&tf);
	QueryPerformanceCounter(&tb);

	for (i = 0; i < n - 1; i++) {
		for (j = 0; j < n - i - 1; j++) {
			if (b[j] > b[j + 1]) {
				swap(b[j], b[j + 1]);
			}
		}
	}

	QueryPerformanceCounter(&te);
	tt = 1000.0*(double(te.QuadPart - tb.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nCPU time: " << tt << " ms\n";
}

void bubbleSortCuda(float *a, float *b, int arraySize)
{
	dim3 threadsPerBlock(arraySize/2);
	dim3 blocksPerGrid(1);

	float *d_a;

	double tt, tu;
	LARGE_INTEGER tb, te, tk, tm, tf;
	QueryPerformanceFrequency(&tf);
	QueryPerformanceCounter(&tb);

	checkCuda(cudaMalloc((void**)&d_a, arraySize * sizeof(float)));
	checkCuda(cudaMemcpy(d_a, a, arraySize * sizeof(float), cudaMemcpyHostToDevice));

	QueryPerformanceCounter(&tk);
	for (int k = 0; k < (arraySize-1); k++) {
		kernel << <blocksPerGrid, threadsPerBlock >> > (d_a, arraySize, k);
	}
	QueryPerformanceCounter(&tm);

	checkCuda(cudaMemcpy(b, d_a, arraySize * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(d_a);

	QueryPerformanceCounter(&te);
	tu = 1000.0*(double(tm.QuadPart - tk.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU 1 time: " << tu << " ms\n";
	tt = 1000.0*(double(te.QuadPart - tb.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU 1 time with memory operations: " << tt << " ms\n";
}

void bubbleSortCuda2(float *a, float *b, int arraySize)
{
	dim3 threadsPerBlock(arraySize / 2);
	dim3 blocksPerGrid(1);

	float *d_a;

	double tt, tu;
	LARGE_INTEGER tb, te, tk, tm, tf;
	QueryPerformanceFrequency(&tf);
	QueryPerformanceCounter(&tb);

	checkCuda(cudaMalloc((void**)&d_a, arraySize * sizeof(float)));
	checkCuda(cudaMemcpy(d_a, a, arraySize * sizeof(float), cudaMemcpyHostToDevice));
	
	QueryPerformanceCounter(&tk);
	kernel2 << <blocksPerGrid, threadsPerBlock >> > (d_a, arraySize);
	QueryPerformanceCounter(&tm);

	checkCuda(cudaMemcpy(b, d_a, arraySize * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(d_a);

	QueryPerformanceCounter(&te);
	tu = 1000.0*(double(tm.QuadPart - tk.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU 2 time: " << tu << " ms\n";
	tt = 1000.0*(double(te.QuadPart - tb.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU 2 time with memory operations: " << tt << " ms\n";
}

void bubbleSortCuda3(float *a, float *b, int arraySize)
{
	dim3 threadsPerBlock(arraySize / 2);
	dim3 blocksPerGrid(1);

	float *d_a;

	double tt, tu;
	LARGE_INTEGER tb, te, tk, tm, tf;
	QueryPerformanceFrequency(&tf);
	QueryPerformanceCounter(&tb);

	checkCuda(cudaMalloc((void**)&d_a, arraySize * sizeof(float)));
	checkCuda(cudaMemcpy(d_a, a, arraySize * sizeof(float), cudaMemcpyHostToDevice));

	QueryPerformanceCounter(&tk);
	kernel3 << <blocksPerGrid, threadsPerBlock, arraySize * sizeof(float) >> > (d_a, arraySize, threadsPerBlock.x);
	QueryPerformanceCounter(&tm);
	
	checkCuda(cudaMemcpy(b, d_a, arraySize * sizeof(float), cudaMemcpyDeviceToHost));
	cudaFree(d_a);

	QueryPerformanceCounter(&te);
	tu = 1000.0*(double(tm.QuadPart - tk.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU 3 time: " << tu << " ms\n";
	tt = 1000.0*(double(te.QuadPart - tb.QuadPart)) / double(tf.QuadPart);
	cout << "\n\nGPU 3 time with memory operations: " << tt << " ms\n";
}

int main()
{
	srand((int)time(nullptr));

	bool display = true;

	const int arraySize = 2048;
    //const int arraySize = 2048;
	float a[arraySize];
	float b[arraySize];

	for (int r = 0; r < arraySize; r++) {
		a[r] = (rand() % 1000) / 100.0;
	}

	if (display) {
		cout << "A" << endl;
		for (int i = 0; i < arraySize; i++) {
			cout << a[i] << "\t";
		}
		cout << endl;
	}

	// CPU ----------------------------------------------------
	for (int r = 0; r < arraySize; r++) {
		b[r] = a[r];
	}

	bubbleSortCpu(b, arraySize);

	if (display) {
		cout << "B" << endl;
		for (int j = 0; j < arraySize; j++) {
			cout << b[j] << "\t";
		}
		cout << endl;
	}

	for (int r = 0; r < arraySize; r++) {
		b[r] = 0;
	}

	// GPU1 ---------------------------------------------------
	bubbleSortCuda(a, b, arraySize);

	if (display) {
		cout << "B" << endl;
		for (int j = 0; j < arraySize; j++) {
			cout << b[j] << "\t";
		}
		cout << endl;
	}

	for (int r = 0; r < arraySize; r++) {
		b[r] = 0;
	}

	// GPU2 ---------------------------------------------------
	bubbleSortCuda2(a, b, arraySize);

	if (display) {
		cout << "B" << endl;
		for (int j = 0; j < arraySize; j++) {
			cout << b[j] << "\t";
		}
		cout << endl;
	}

	for (int r = 0; r < arraySize; r++) {
		b[r] = 0;
	}

	// GPU3 ---------------------------------------------------
	bubbleSortCuda3(a, b, arraySize);

	if (display) {
		cout << "B" << endl;
		for (int j = 0; j < arraySize; j++) {
			cout << b[j] << "\t";
		}
		cout << endl;
	}

	// END ----------------------------------------------------

    checkCuda(cudaDeviceReset());
    
	cout << "DONE";
	cin.ignore();

    return 0;
}
