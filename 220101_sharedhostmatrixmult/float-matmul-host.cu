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

#define WIDTH 1024
#define TILE_WIDTH 32
#define WARPSIZE TILE_WIDTH
#define GRIDSIZE ((WIDTH/TILE_WIDTH)*(WIDTH/TILE_WIDTH))
#define BLOCKSIZE (TILE_WIDTH*TILE_WIDTH)
#define TOTALSIZE (GRIDSIZE*BLOCKSIZE)

void genData(float* ptr, unsigned int size)
{
	while (size--) {
		*ptr++ = (float)(rand() % 1000) / 1000.0F;
	}
}

void matmulti_host(const float* g_A, const float* g_B, float* g_C, const int width) {
	for (register int gy = 0; gy < width; gy++) {
		for (register int gx = 0; gx < width; gx++) {
			register float sum = 0.0F;
			for (register int k = 0; k < width; k++) {
				sum += g_A[gy * width + k] * g_B[k * width + gx];
			}
			g_C[gy * width + gx] = sum;
		}
	}
}

__host__ int main(void)
{
	//바로 아래 두 줄은 QueryPerformance로 CUDA이벤트 쓸때는 
	//float형으로 아래의 메모리할당을 해주어야 오류가 안 나길래 적었습니다
	float* pSource = NULL;
	float* pResult = NULL;
	long long cntStart, cntEnd, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)(&freq));
	pSource = (float*)malloc(sizeof(float) * TOTALSIZE);
	pResult = (float*)malloc(sizeof(float) * TOTALSIZE);

	//host변수 선언 및 초기화
	float* pA = NULL;
	float* pB = NULL;
	float* pC = NULL;
	pA = (float*)malloc(sizeof(float) * TOTALSIZE);
	pB = (float*)malloc(sizeof(float) * TOTALSIZE);
	pC = (float*)malloc(sizeof(float) * TOTALSIZE);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	genData(pA, TOTALSIZE);
	genData(pB, TOTALSIZE);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	//printf("elasped time : %f usec\n", (double)(cntEnd - cntStart) * 1000000.0 / (double)(freq));
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	matmulti_host(pA, pB, pC, WIDTH);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 1000.0 / (double)freq);
	//마지막에 host변수 메모리해제코드
	free(pSource);
	free(pResult);
	free(pA);
	free(pB);
	free(pC);
}