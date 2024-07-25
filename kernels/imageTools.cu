#include <stdio.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "imageTools.h"
#include "stb_image_write.h"
#include <vector>
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess)
  {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void imageLoad(unsigned char *image, uchar4 *imageLoaded, size_t imgSize)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < imgSize; i += stride)
  {
    imageLoaded[i].x = image[i * 4 + 0];
    imageLoaded[i].y = image[i * 4 + 1];
    imageLoaded[i].z = image[i * 4 + 2];
    imageLoaded[i].w = image[i * 4 + 3];
  }
}

__global__ void imageWrite(unsigned char *image, uchar4 *pixels, int width, int height)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < width * height; i += stride)
  {
    image[i * 4 + 0] = pixels[i].x;
    image[i * 4 + 1] = pixels[i].y;
    image[i * 4 + 2] = pixels[i].z;
    image[i * 4 + 3] = pixels[i].w;
  }
}

void imageWriteWrapper(const char *filename, uchar4 *pixels, int width, int height)
{
  unsigned char *image;
  uchar4 *d_pixels;

  checkCuda(cudaMallocManaged(&image, width * height * 4 * sizeof(unsigned char)));
  checkCuda(cudaMallocManaged(&d_pixels, width * height * sizeof(uchar4)));
  checkCuda(cudaMemcpy(d_pixels, pixels, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int numBlocks = (width * height + threadsPerBlock - 1) / threadsPerBlock;

  imageWrite<<<numBlocks, threadsPerBlock>>>(image, d_pixels, width, height);
  checkCuda(cudaGetLastError());
  cudaDeviceSynchronize();
  unsigned char *d_image = (unsigned char *)malloc(width * height * 4 * sizeof(unsigned char));
  cudaMemcpy(d_image, image, width * height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  stbi_write_png(filename, width, height, 4, d_image, width * 4);

  cudaFree(d_pixels);
  cudaFree(image);
}

void imageLoadWrapper(unsigned char *image, uchar4 *imageLoaded, size_t imgSize)
{
  unsigned char *d_image;
  uchar4 *d_imageLoaded;

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

__global__ void imageGrayScale(uchar4 *returnImage, uchar4 *imageLoaded, size_t imgSize)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < imgSize; i += stride)
  {
    uchar4 pixel = imageLoaded[i];
    unsigned char gray = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    returnImage[i].x = gray;
    returnImage[i].y = gray;
    returnImage[i].z = gray;
    returnImage[i].w = pixel.w;
  }
}

void imageGrayScaleWrapper(uchar4 *returnImage, uchar4 *imageLoaded, size_t imgSize)
{
  uchar4 *d_returnImage;
  uchar4 *d_imageLoaded;

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

__global__ void imageSobelEdge(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
  {
    float Gx = 0;
    float Gy = 0;

    int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int ky = -1; ky <= 1; ky++)
    {
      for (int kx = -1; kx <= 1; kx++)
      {
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

__global__ void imageGaussianBlur(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height, int kernalSize, float *kernal)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
  {
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (int ky = -kernalSize / 2; ky <= kernalSize / 2; ky++)
    {
      for (int kx = -kernalSize / 2; kx <= kernalSize / 2; kx++)
      {
        int idx = (y + ky) * width + (x + kx);
        uchar4 pixel = imageLoaded[idx];
        float kernalValue = kernal[(ky + kernalSize / 2) * kernalSize + (kx + kernalSize / 2)];
        sum.x += pixel.x * kernalValue;
        sum.y += pixel.y * kernalValue;
        sum.z += pixel.z * kernalValue;
      }
    }
    returnImage[y * width + x].x = sum.x;
    returnImage[y * width + x].y = sum.y;
    returnImage[y * width + x].z = sum.z;
    returnImage[y * width + x].w = 255;
  }
}

__global__ void imageMeanBlur(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height, int kernalSize)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
  {
    int3 mean = make_int3(0, 0, 0);
    for (int ky = -kernalSize / 2; ky <= kernalSize / 2; ky++)
    {
      for (int kx = -kernalSize / 2; kx <= kernalSize / 2; kx++)
      {
        int idx = (y + ky) * width + (x + kx);
        uchar4 pixel = imageLoaded[idx];
        mean.x += pixel.x;
        mean.y += pixel.y;
        mean.z += pixel.z;
      }
    }
    
    mean.x = (mean.x/(kernalSize*kernalSize));
    mean.y = (mean.y/(kernalSize*kernalSize));
    mean.z = (mean.z/(kernalSize*kernalSize));
    returnImage[y * width + x].x = mean.x;
    returnImage[y * width + x].y = mean.y;
    returnImage[y * width + x].z = mean.z;
    returnImage[y * width + x].w = 255;
  }
}

void imageSobelEdgeWrapper(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height)
{
  uchar4 *d_returnImage;
  uchar4 *d_imageLoaded;
  checkCuda(cudaMallocManaged(&d_returnImage, width * height * sizeof(uchar4)));
  checkCuda(cudaMallocManaged(&d_imageLoaded, width * height * sizeof(uchar4)));
  checkCuda(cudaMemcpy(d_imageLoaded, imageLoaded, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  imageSobelEdge<<<blocks, threads>>>(d_returnImage, d_imageLoaded, width, height);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  cudaMemcpy(returnImage, d_returnImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_returnImage);
  cudaFree(d_imageLoaded);
}

float *generateGaussianKernal(int size, float sigma)
{
  float *kernal = (float *)malloc(size * size * sizeof(float));
  float sum = 0.0f;
  int halfSize = size / 2;
  for (int i = -halfSize; i <= halfSize; i++)
  {
    for (int j = -halfSize; j <= halfSize; j++)
    {
      float value = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
      kernal[(i + halfSize) * size + (j + halfSize)] = value;
      sum += value;
    }
  }

  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      kernal[i * size + j] /= sum;
    }
  }
  return kernal;
}

void imageGaussianBlurWrapper(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height, int size, float sigma)
{
  float *kernal = generateGaussianKernal(size, sigma);
  uchar4 *d_returnImage;
  uchar4 *d_imageLoaded;
  float *d_kernal;
  checkCuda(cudaMallocManaged(&d_returnImage, width * height * sizeof(uchar4)));
  checkCuda(cudaMallocManaged(&d_imageLoaded, width * height * sizeof(uchar4)));
  checkCuda(cudaMallocManaged(&d_kernal, size * size * sizeof(float)));
  checkCuda(cudaMemcpy(d_imageLoaded, imageLoaded, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_kernal, kernal, size * size * sizeof(float), cudaMemcpyHostToDevice));
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  imageGaussianBlur<<<blocks, threads>>>(d_returnImage, d_imageLoaded, width, height, size, d_kernal);
  checkCuda(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(returnImage, d_returnImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_returnImage);
  cudaFree(d_imageLoaded);
  cudaFree(d_kernal);
  free(kernal);
}

void imageMeanBlurWrapper(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height)
{
  uchar4 *d_image_return;
  uchar4 *d_image_loaded;
  checkCuda(cudaMalloc(&d_image_loaded, width * height * sizeof(uchar4)));
  checkCuda(cudaMalloc(&d_image_return, width * height * sizeof(uchar4)));
  checkCuda(cudaMemcpy(d_image_loaded, imageLoaded, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  imageMeanBlur<<<blocks, threads>>>(d_image_return, d_image_loaded, width, height, 3);
  checkCuda(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaMemcpy(returnImage, d_image_return, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_image_return);
  cudaFree(d_image_loaded);
}
