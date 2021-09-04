#include<cstdio>
#include "device_launch_parameters.h"
#include "cuda_runtime.h""
#include<stdio.h>
#include<stdlib.h>//rand()
#include<windows.h>//performance counter
#include<time.h>
#include<WinBase.h>
#include<io.h>
#include<sys/stat.h>
#include<chrono>
#include<cstring>
#include<string.h>

#define WIDTH 4
#define TILE_WIDTH 2
#define BLOCKSIZE (TILE_WIDTH*TILE_WIDTH)
#define GRIDSIZE (2*2)
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

//#if defined(NDEBUG)
//#define CUDA_CHECK(x)	(x)
// code for release mode
//#else
// code for debug mode debug mode is defined(_DEBUG)
//#define CUDA_CHECK(x)	do{\
//	(x);\
//	cudaError_t e = cudaGetLastError();\
//	if(cudaSuccess!=e){\
//		printf("cuda failure \"%s\" at %s:%d\n", \
//			cudaGetErrorString(e),\
//			__FILE__, __LINE__);\
//		exit(1);\
//	}\
//	}while(0)
//#endif

//#define CUDA_CHECK()
//	cudaError_t e = cudaGetLastError();\
//	if(cudaSuccess!=e){\
//		printf("cuda failure \"%s\" at %s:%d\n", \
//			cudaGetErrorString(e),\
//			__FILE__, __LINE__);\
//		exit(1);\
//	}
// no ; needed after CUDA_CHECK()

#define CUDA_CHECK() do{\
	cudaError_t e = cudaGetLastError();\
	if(cudaSuccess!=e){\
		printf("cuda failure \"%s\" at %s:%d\n", \
			cudaGetErrorString(e),\
			__FILE__, __LINE__);\
		exit(1);\
	}\
}while(0)

using namespace std;
using namespace chrono;

void genData(unsigned* pData, int size)
{
	while (--size) {
		*pData++ = (unsigned)(rand() % 10);
	}
}
__global__ void matmul(unsigned* g_C, const unsigned* g_A, const unsigned* g_B, int width)
{
	unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int gy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	unsigned sum = 0;
	printf("%d번째 g_C를 구하는 과정입니다\n", gy * width + gx);
	for (int m = 0; m < TILE_WIDTH; m++) {//TILE_WIDTH = blockDim.y(g_A의 경우)= blockDim.x(g_B의 경우)
		printf("g_A[%4d]*g_B[%4d] = %u * %u = %u더하고\n", gy*width+(m*TILE_WIDTH+tx), (m*TILE_WIDTH+ty)*width+gx, g_A[gy*width+(m*TILE_WIDTH+tx)]*g_B[(m*TILE_WIDTH+ty)*width+gx], g_A[gy * width + (m * TILE_WIDTH + tx)], g_B[(m * TILE_WIDTH + ty) * width + gx]);
		sum += g_A[gy * width + (m * TILE_WIDTH + tx)]*g_B[(m*TILE_WIDTH+ty)*width+gx];

	}
	g_C[gy * width + gx] = sum;
	printf("결과는 g_C[% 4d][% 4d] = % u\n", gy,gx,g_C[gy*width+gx]);
}
int main(void)
{
	unsigned* pA = NULL;
	unsigned* pB = NULL;
	unsigned* pC = NULL;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	pA = (unsigned*)malloc(sizeof(unsigned) * TOTALSIZE);
	pB = (unsigned*)malloc(sizeof(unsigned) * TOTALSIZE);
	pC = (unsigned*)malloc(sizeof(unsigned) * TOTALSIZE);
	genData(pA, TOTALSIZE); genData(pB, TOTALSIZE);
	printf("pA = {%u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u}\n", pA[0], pA[1], pA[2], pA[3], pA[4], pA[5], pA[6], pA[7], pA[8], pA[9], pA[10], pA[11], pA[12], pA[13], pA[14], pA[15]);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("cuda failure \"%s\" at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(1);
	}
	else {
		printf("cuda success at %s:%d\n", __FILE__, __LINE__);
	}

	printf("pB = {%u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u}\n", pB[0], pB[1], pB[2], pB[3], pB[4], pB[5], pB[6], pB[7], pB[8], pB[9], pB[10], pB[11], pB[12], pB[13], pB[14], pB[15]);
	err = cudaGetLastError();
	if (cudaSuccess != err) {
		printf("cuda failure \"%s\" at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);
		exit(1);
	}
	else {
		printf("cuda success at %s:%d\n", __FILE__, __LINE__);
	}

	unsigned* pADev = 0;
	unsigned* pBDev = 0;
	unsigned* pCDev = 0;

	cudaMalloc((void**)&pADev, TOTALSIZE * sizeof(unsigned));
	CUDA_CHECK();
	(cudaMalloc((void**)&pBDev, TOTALSIZE * sizeof(unsigned)));
	CUDA_CHECK();
	(cudaMalloc((void**)&pCDev, TOTALSIZE * sizeof(unsigned)));
	CUDA_CHECK();

	(cudaMemset(pADev, 0, TOTALSIZE*sizeof(unsigned)));
	CUDA_CHECK();
	(cudaMemset(pBDev, 0, TOTALSIZE*sizeof(unsigned)));
	CUDA_CHECK();
	(cudaMemset(pCDev, 0, TOTALSIZE*sizeof(unsigned)));
	CUDA_CHECK();

	cudaMemcpy(pADev, pA, sizeof(unsigned) * TOTALSIZE, cudaMemcpyHostToDevice);
	CUDA_CHECK();
	(cudaMemcpy(pBDev, pB, sizeof(unsigned) * TOTALSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK();

	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));

	dim3 gridDim(WIDTH/TILE_WIDTH,WIDTH/TILE_WIDTH, 1);
	dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
	matmul<<<gridDim,blockDim>>>(pCDev, pADev, pBDev, WIDTH);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	(cudaPeekAtLastError());
	CUDA_CHECK();
	(cudaMemcpy(pC, pCDev, sizeof(unsigned) * TOTALSIZE, cudaMemcpyDeviceToHost));
	CUDA_CHECK();
	for (int row = 0; row < WIDTH; row++) {
		for (int col = 0; col < WIDTH; col++) {
			printf("pCDev[%4d][%4d]=%u\n", row, col, pC[row * WIDTH + col]);
		}
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("cuda failure at \"%s\" \n", cudaGetErrorString(err));
	}
	else {
		printf("cuda success at cuda function\n");
	}
	for (int row = 0; row < WIDTH; row++) {
		for (int col = 0; col < WIDTH; col++) {
			printf("pC[%4d][%4d]=%u\n", row, col, pC[row * WIDTH + col]);
		}
	}
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("cuda failure at \"%s\" \n", cudaGetErrorString(err));
	}
	else {
		printf("cuda success at memcpyDeviceToHost\n");
	}
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 1000.0 / freq);
	fflush(stdout);
	free(pA);
	free(pB);
	free(pC);
	(cudaFree(pADev));
	CUDA_CHECK();
	(cudaFree(pBDev));
	CUDA_CHECK();
	(cudaFree(pCDev));
	CUDA_CHECK();
	return 0;
}