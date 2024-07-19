#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "imageLoad.h"

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void imageLoad(unsigned char* image, uchar4* imageLoaded, size_t imgSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i <imgSize) {
        imageLoaded[i].x = image[i * 4 + 0];
        imageLoaded[i].y = image[i * 4 + 1];
        imageLoaded[i].z = image[i * 4 + 2];
        imageLoaded[i].w = image[i * 4 + 3];
    }

}

void imageLoadWrapper(unsigned char* image, uchar4* imageLoaded, size_t imgSize) {
    unsigned char* d_image;
    uchar4* d_imageLoaded;

    checkCuda(cudaMallocManaged(&d_image, imgSize * 4 * sizeof(unsigned char)));
    checkCuda(cudaMallocManaged(&d_imageLoaded, imgSize * sizeof(uchar4)));
    checkCuda(cudaMemcpy(d_image, image, imgSize * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    int imgSizeInt = (int)imgSize;
    int threadsPerBlock = 256;
    int numBlocks = (imgSizeInt + threadsPerBlock - 1) / threadsPerBlock;
    imageLoad<<<numBlocks, threadsPerBlock>>>(d_image, d_imageLoaded, imgSize);
    checkCuda(cudaGetLastError());
    cudaDeviceSynchronize();
    for(int i=0;i<imgSize;i++) {
        printf("%hhu\n", d_image[i].x);
    }
    cudaFree(d_image);
    cudaFree(d_imageLoaded);
}