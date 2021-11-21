

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>
#include<string.h>
#include<malloc.h>
#include<windows.h>
#include<time.h>
#include<Winbase.h>
#include<cstring>
#include<cstdio>
#include<math.h>
#include<io.h>
#include<fcntl.h>
#include<sys/stat.h>
#include<chrono>
#include<crt/device_functions.h>

#if defined(NDEBUG)
#define CUDA_CHECK(x)	(x)
#else
#define CUDA_CHECK(x)	do{\
	(x);\
	cudaError_t e = cudaGetLastError();\
	if(cudaSuccess!=e){\
		printf("cuda failure \"%s\" at %s:%d\n", \
			cudaGetErrorString(e),\
			__FILE__, __LINE__);\
		exit(1);\
	}\
}while(0)
#endif

using namespace std;
using namespace chrono;
typedef duration<long long, nano> nanoseconds;
typedef duration<long long, micro> microsecons;
typedef duration<long long, milli> milliseconds;

#define WIDTH 8
#define TILE_WIDTH 2
#define WARPSIZE TILE_WIDTH
#define GRIDSIZE ((WIDTH/TILE_WIDTH)*(WIDTH/TILE_WIDTH))
#define BLOCKSIZE (TILE_WIDTH*TILE_WIDTH)
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

void genDataA(unsigned* ptr, int size)
{
	unsigned int gy = 0, gx = 0;
	for (unsigned int gy = 0; gy < WIDTH; gy++) {
		for (unsigned int gx = 0; gx < WIDTH; gx++) {
			*ptr++ = gy * WIDTH + gx;
		}
	}
}
void genDataB(unsigned* ptr, int size)
{
	unsigned int gy = 0, gx = 0;
	for (unsigned int gy = 0; gy < WIDTH; gy++) {
		for (unsigned int gx = 0; gx < WIDTH; gx++) {
			*ptr++ = (gx * WIDTH + gy);
		}
	}
}
void printData(unsigned* ptr, int size)
{
	for (unsigned int gy = 0; gy < WIDTH; gy++) {
		for (unsigned int gx = 0; gx < WIDTH; gx++) {
			printf("%-3d ", ptr[gy * WIDTH + gx]);
		}
		printf("\n");
	}
}
//width에 보내줄 변수는 for문에서 k의 종결조건으로 사용할 A의 총 열의 개수 혹은 B의 총 행의 개수
//width는 WIDTH
void matmulti_host(const unsigned* g_A, const unsigned* g_B, unsigned* g_C, const int width) {
	for (register int gy = 0; gy < width; gy++) {
		for (register int gx = 0; gx < width; gx++) {
			register unsigned sum = 0;
			for (register int k = 0; k < width; k++) {
				sum += g_A[gy * width + k] * g_B[k * width + gx];
			}
			g_C[gy * width + gx] = sum;
		}
	}
}
//width에 보내줄 변수는 for문에서 k의 종결조건으로 사용할 A의 총 열의 개수 혹은 B의 총 행의 개수
//width는 WIDTH
__global__ void matmulti_shared(const unsigned* g_A, const unsigned* g_B, unsigned* g_C, const int width)
{
	__shared__ unsigned s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ unsigned s_B[TILE_WIDTH][TILE_WIDTH];
	int by = blockIdx.y, bx = blockIdx.x;
	int ty = threadIdx.y, tx = threadIdx.x;
	int gy = by * TILE_WIDTH + ty;
	int gx = bx * TILE_WIDTH + tx;
	unsigned sum = 0;
	for (register int m = 0; m < width / TILE_WIDTH; m++) {
		//global memory에서 shared memory로 가져올 때 gy,gx는 고정해두고
		s_A[ty][tx] = g_A[gy * width + (m * TILE_WIDTH + tx)];
		s_B[ty][tx] = g_B[(m * TILE_WIDTH + ty) * width + gx];
		__syncthreads();
		//k는 shared memory들로 g_C의 성분을 구하기 위한 것이므로
		//위의 설명에 따라 k는 TILE_WIDTH가 종결조건입니다
		for (register int k = 0; k < TILE_WIDTH; k++) {
			sum += s_A[ty][k] * s_B[k][tx];
		}
		__syncthreads();
	}
	g_C[gy * width + gx] = sum;
}
__host__ int main(void)
{
	//바로 아래 두 줄은 QueryPerformance로 CUDA이벤트 쓸때는 
	//float형으로 아래의 메모리할당을 해주어야 오류가 안 나길래 적었습니다
	float* pSource = NULL;
	float* pResult = NULL;
	//시간변수선언
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	pSource = (float*)malloc(sizeof(float) * TOTALSIZE);
	pResult = (float*)malloc(sizeof(float) * TOTALSIZE);

	//host변수 선언 및 초기화
	unsigned* pA = NULL;
	unsigned* pB = NULL;
	unsigned* pC = NULL;
	pA = (unsigned*)malloc(sizeof(unsigned) * TOTALSIZE);
	pB = (unsigned*)malloc(sizeof(unsigned) * TOTALSIZE);
	pC = (unsigned*)malloc(sizeof(unsigned) * TOTALSIZE);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	genDataA(pA, TOTALSIZE);
	genDataB(pB, TOTALSIZE);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("elasped time : %f usec\n", (double)(cntEnd - cntStart) * 10000.0 / (double)(freq));
	printf("g_A component;\n");
	printData(pA, TOTALSIZE);
	printf("g_B component;\n");
	printData(pB, TOTALSIZE);
	//device변수 선언
	unsigned* pADev = NULL;
	unsigned* pBDev = NULL;
	unsigned* pCDev = NULL;
	//cudaMalloc,cudaMemset, cudaMemcpy:H->D
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	CUDA_CHECK(cudaMalloc((void**)&pADev, sizeof(unsigned) * TOTALSIZE));//WIDTH*WIDTH
	CUDA_CHECK(cudaMemset(pADev, 0, sizeof(unsigned) * TOTALSIZE));
	CUDA_CHECK(cudaMemcpy(pADev, pA, sizeof(unsigned) * TOTALSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void**)&pBDev, sizeof(unsigned) * TOTALSIZE));//WIDTH*WIDTH
	CUDA_CHECK(cudaMemset(pBDev, 0, sizeof(unsigned) * TOTALSIZE));
	CUDA_CHECK(cudaMemcpy(pBDev, pB, sizeof(unsigned) * TOTALSIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMalloc((void**)&pCDev, sizeof(unsigned) * TOTALSIZE));//WIDTH*WIDTH
	CUDA_CHECK(cudaMemset(pCDev, 0, sizeof(unsigned) * TOTALSIZE));
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("elapsed time for 디바이스변수 초기화= %f msec\n", (double)(cntEnd - cntStart) * 100000.0 / (double)(freq));
	//cuda kernel 시간측정시작
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	dim3 dimGrid(GRIDSIZE, 1, 1);
	dim3 dimBlock(BLOCKSIZE, 1, 1);
	matmulti_shared<<<dimGrid,dimBlock>>>(pADev, pBDev, pCDev, WIDTH);
	//cuda kernel 시간측정 종료
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	CUDA_CHECK(cudaPeekAtLastError());
	//cudaMemcpy:D->H, host변수 출력
	CUDA_CHECK(cudaMemcpy(pC, pCDev, sizeof(unsigned) * TOTALSIZE, cudaMemcpyDeviceToHost));
	for (unsigned int y = 0; y < WIDTH; y++) {
		for (unsigned int x = 0; x < WIDTH; x++) {
			printf("g_C[%4d][%4d] = %u\n", y, x, pC[y * WIDTH + x]);
		}
		printf("\n");
	}
	//시간출력
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 100000.0 / (double)(freq));
	printData(pC, GRIDSIZE);
	fflush(stdout);
	//host변수 메모리해제
	free(pSource);
	free(pResult);
	free(pA);
	free(pB);
	free(pC);
	//cudaFree
	CUDA_CHECK(cudaFree(pADev));
	CUDA_CHECK(cudaFree(pBDev));
	CUDA_CHECK(cudaFree(pCDev));
	//fflush, return 0
	fflush(stdout);
	return 0;
}