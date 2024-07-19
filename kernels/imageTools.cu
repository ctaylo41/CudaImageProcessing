#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "imageTools.h"
#include "stb_image_write.h"
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void imageLoad(unsigned char* image, uchar4* imageLoaded, size_t imgSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i <imgSize;i+=stride) {
        imageLoaded[i].x = image[i * 4 + 0];
        imageLoaded[i].y = image[i * 4 + 1];
        imageLoaded[i].z = image[i * 4 + 2];
        imageLoaded[i].w = image[i * 4 + 3];
    }
}


__global__ void imageWrite(unsigned char* image, uchar4* pixels, int width, int height) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for(int i = index; i < width * height; i+=stride) {
      image[i * 4 + 0] = pixels[i].x; 
      image[i * 4 + 1] = pixels[i].y; 
      image[i * 4 + 2] = pixels[i].z; 
      image[i * 4 + 3] = pixels[i].w;
  }
}

void imageWriteWrapper(const char* filename, uchar4* pixels,int width,int height) {
    unsigned char* image;
    uchar4* d_pixels;
    
    checkCuda(cudaMallocManaged(&image, width * height * 4 * sizeof(unsigned char)));
    checkCuda(cudaMallocManaged(&d_pixels, width * height * sizeof(uchar4)));
    checkCuda(cudaMemcpy(d_pixels, pixels, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    
    imageWrite<<<numBlocks, threadsPerBlock>>>(image, d_pixels, width, height);
    checkCuda(cudaGetLastError());
    cudaDeviceSynchronize();
    unsigned char* d_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));
    cudaMemcpy(d_image, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_png(filename, width, height, 4, d_image, width * 4);
    
    cudaFree(d_pixels);
    cudaFree(image);
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
    cudaMemcpy(imageLoaded, d_imageLoaded, imgSize * sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_imageLoaded);
}

__global__ void imageGrayScale(uchar4* returnImage, uchar4* imageLoaded, size_t imgSize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < imgSize; i+=stride) {
        uchar4 pixel = imageLoaded[i];
        unsigned char gray = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
        returnImage[i].x = gray;
        returnImage[i].y = gray;
        returnImage[i].z = gray;
        returnImage[i].w = pixel.w;
    }
}

void imageGrayScaleWrapper(uchar4* returnImage, uchar4* imageLoaded, size_t imgSize) {
    uchar4* d_returnImage;
    uchar4* d_imageLoaded;

    checkCuda(cudaMallocManaged(&d_returnImage, imgSize * sizeof(uchar4)));
    checkCuda(cudaMallocManaged(&d_imageLoaded, imgSize * sizeof(uchar4)));
    checkCuda(cudaMemcpy(d_imageLoaded, imageLoaded, imgSize * sizeof(uchar4), cudaMemcpyHostToDevice));

    int imgSizeInt = (int)imgSize;
    int threadsPerBlock = 256;
    int numBlocks = (imgSizeInt + threadsPerBlock - 1) / threadsPerBlock;
    imageGrayScale<<<numBlocks, threadsPerBlock>>>(d_returnImage, d_imageLoaded, imgSize);
    checkCuda(cudaGetLastError());
    cudaDeviceSynchronize();
    cudaMemcpy(returnImage, d_returnImage, imgSize * sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_returnImage);
    cudaFree(d_imageLoaded);
}

__global__ void imageSobelEdge(uchar4* returnImage, uchar4* imageLoaded, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int stride = blockDim.x * gridDim.x;
  if(x > 0 && y > 0 && x < width - 1 && y < height - 1) {
    float Gx = 0;
    float Gy = 0;

    int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for(int ky = -1; ky <= 1; ky++) {
      for(int kx = -1; kx <= 1; kx++) {
        int idx = (y + ky) * width + (x + kx);
        uchar4 pixel = imageLoaded[idx];
        Gx += pixel.x * sobelX[ky + 1][kx + 1];
        Gy += pixel.x * sobelY[ky + 1][kx + 1];
      }
    }
    float magnitude = sqrt(Gx * Gx + Gy * Gy);
    magnitude = max(0.0f, min(255.0f, magnitude));
    returnImage[y * width + x].x = magnitude;
    returnImage[y * width + x].y = magnitude;
    returnImage[y * width + x].z = magnitude;
    returnImage[y * width + x].w = 255;
    
  }
}

void imageSobelEdgeWrapper(uchar4* returnImage, uchar4* imageLoaded, size_t imgSize) {
    uchar4* d_returnImage;
    uchar4* d_imageLoaded;

    checkCuda(cudaMallocManaged(&d_returnImage, imgSize * sizeof(uchar4)));
    checkCuda(cudaMallocManaged(&d_imageLoaded, imgSize * sizeof(uchar4)));
    checkCuda(cudaMemcpy(d_imageLoaded, imageLoaded, imgSize * sizeof(uchar4), cudaMemcpyHostToDevice));
    int imgSizeInt = (int)imgSize;
    int threadsPerBlock = 16;
    int numBlocks = (imgSizeInt + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocks(numBlocks, numBlocks);
    dim3 threads(threadsPerBlock, threadsPerBlock);
    printf("here\n");

    imageSobelEdge<<<blocks, threads>>>(d_returnImage, d_imageLoaded, 512, 512);
    
    checkCuda(cudaGetLastError());

    cudaDeviceSynchronize();
    cudaMemcpy(returnImage, d_returnImage, imgSize * sizeof(uchar4), cudaMemcpyDeviceToHost);
    cudaFree(d_returnImage);
    cudaFree(d_imageLoaded);
}