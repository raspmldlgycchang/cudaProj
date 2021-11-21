
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
			printf("%-7d ", ptr[gy * WIDTH + gx]);
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
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntStart));
	matmulti_host(pA, pB, pC, WIDTH);
	QueryPerformanceCounter((LARGE_INTEGER*)(&cntEnd));
	printf("elapsed time = %f msec\n", (double)(cntEnd - cntStart) * 100000.0 / (double)freq);
	printData(pC, GRIDSIZE);
	//마지막에 host변수 메모리해제코드
	free(pSource);
	free(pResult);
	free(pA);
	free(pB);
	free(pC);
}