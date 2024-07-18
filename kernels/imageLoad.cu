#include <stdio.h>
#include "imageLoad.h"
__global__ void imageLoad(unsigned char* image, uchar4* imageLoaded, size_t imgSize) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; i < imgSize; i += stride) {
        imageLoaded[i].x = image[i * 4 + 0];
        imageLoaded[i].y = image[i * 4 + 1];
        imageLoaded[i].z = image[i * 4 + 2];
        imageLoaded[i].w = image[i * 4 + 3];
    }

}

void imageLoadWrapper(unsigned char* image, uchar4* imageLoaded, size_t imgSize) {
    cudaMallocManaged(&image, imgSize * sizeof(unsigned char));
    cudaMallocManaged(&imageLoaded, imgSize * sizeof(uchar4));
    int blockSize = 256;
    int numBlocks = (imgSize + blockSize - 1) / blockSize;
    imageLoad<<<numBlocks, blockSize>>>(image, imageLoaded, imgSize);
    cudaDeviceSynchronize();
    cudaFree(image);
    cudaFree(imageLoaded);
}