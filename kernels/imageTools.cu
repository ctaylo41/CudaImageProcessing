#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include "imageTools.h"
#include "stb_image_write.h"
#include <vector>
#include <cfloat>
#include <float.h>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cufft.h>


#define TOLERANCE 1e-4

struct Complex
{
  float real;
  float imag;

  __host__ __device__ Complex(float r = 0.0f, float i = 0.0f) : real(r), imag(i) {}

  __host__ __device__ Complex operator+(const Complex &b) const
  {
    return Complex{real + b.real, imag + b.imag};
  }

  __host__ __device__ Complex operator-(const Complex &b) const
  {
    return Complex{real - b.real, imag - b.imag};
  }

  __host__ __device__ Complex operator*(const Complex &b) const
  {
    return Complex{real * b.real - imag * b.imag, real * b.imag + imag * b.real};
  }

  __host__ __device__ float magnitude() const
  {
    return sqrt(real * real + imag * imag);
  }

  __host__ __device__ Complex operator/(float scalar) const
  {
    return Complex(real / scalar, imag / scalar);
  }
};

struct ComplexRGB
{
  Complex r;
  Complex g;
  Complex b;

  __device__ __host__ ComplexRGB() : r(0, 0), g(0, 0), b(0, 0) {}
  __device__ __host__ ComplexRGB(Complex r, Complex g, Complex b) : r(r), g(g), b(b) {}

  __device__ __host__ ComplexRGB operator+(const ComplexRGB &Other) const
  {
    return ComplexRGB{r + Other.r, g + Other.g, b + Other.b};
  }

  __device__ __host__ ComplexRGB operator-(const ComplexRGB &Other) const
  {
    return ComplexRGB{r - Other.r, g - Other.g, b - Other.b};
  }

  __device__ __host__ ComplexRGB operator*(const ComplexRGB &Other) const
  {
    return ComplexRGB{r * Other.r, g * Other.g, b * Other.b};
  }

  __device__ __host__ ComplexRGB operator*(const float &scalar) const
  {
    return ComplexRGB{r * scalar, g * scalar, b * scalar};
  }

  __device__ __host__ float3 magnitude() const
  {
    return float3{r.magnitude(), g.magnitude(), b.magnitude()};
  }

  __device__ __host__ ComplexRGB operator/(float scalar)
  {
    return ComplexRGB(r / scalar, g / scalar, b / scalar);
  }
};
bool compareComplexRGB(const ComplexRGB &a, const ComplexRGB &b)
{
  return (std::fabs(a.r.real - b.r.real) < TOLERANCE) &&
         (std::fabs(a.g.real - b.g.real) < TOLERANCE) &&
         (std::fabs(a.b.real - b.b.real) < TOLERANCE);
}
cudaError_t checkCuda(cudaError_t result)
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
    printf("allpocated\n");

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
  cudaError_t cudaStatus = cudaMallocManaged(&d_image, imgSize * 4 * sizeof(unsigned char));
  if (cudaStatus == cudaSuccess)
  {
    printf("malloced\n");
  }
  else
  {
    fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
  }

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

struct rgbCufft {
  cufftComplex* red;
  cufftComplex* blue;
  cufftComplex* green;
};

__device__ void printImage(Complex *data, int width, int height)
{
  
  for(int i=0;i<width*height;i++) {
    printf("(%.4f,%.4f) ",data[i].real,data[i].imag);
    if((i+1)%width==0) {
      printf("\n");
    }
  }
}


__device__ void printImage(rgbCufft data,int width, int height) {
  for(int i = 0;i<width*height;i++) {
    printf("(%.4f,%.4f,%.4f)\n", data.red[i].x, data.green[i].x, data.blue[i].x);
  }
}

__global__ void printImageKernel(Complex *data, int width, int height) {
    printImage(data, width, height);
}

__global__ void printImageKernel(rgbCufft data, int width, int height) {
    printImage(data,width,height);
}

__global__ void imageGrayScale(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height)
  {
    int idx = width * y + x;
    uchar4 pixel = imageLoaded[idx];
    unsigned char gray = (unsigned char)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
    returnImage[idx].x = gray;
    returnImage[idx].y = gray;
    returnImage[idx].z = gray;
    returnImage[idx].w = pixel.w;
  }
}

void imageGrayScaleWrapper(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height)
{
  uchar4 *d_returnImage;
  uchar4 *d_imageLoaded;
  int imgSize = height * width;
  checkCuda(cudaMallocManaged(&d_returnImage, imgSize * sizeof(uchar4)));
  checkCuda(cudaMallocManaged(&d_imageLoaded, imgSize * sizeof(uchar4)));
  checkCuda(cudaMemcpy(d_imageLoaded, imageLoaded, imgSize * sizeof(uchar4), cudaMemcpyHostToDevice));

  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  imageGrayScale<<<blocks, threads>>>(d_returnImage, d_imageLoaded, width, height);
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

    mean.x = (mean.x / (kernalSize * kernalSize));
    mean.y = (mean.y / (kernalSize * kernalSize));
    mean.z = (mean.z / (kernalSize * kernalSize));
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


__device__ float atomicMinFloat(float *address, float val)
{
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

__device__ float atomicMaxFloat(float *address, float val)
{
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;

  do
  {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);

  return __int_as_float(old);
}

__global__ void findMinMax(float *image, int width, int height, float *min, float *max)
{
  extern __shared__ float sharedData[];
  float *sMin = sharedData;
  float *sMax = sharedData + blockDim.x * blockDim.y;

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = y * width + x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;

  float localMin = FLT_MIN;
  float localMax = -FLT_MAX;

  if (x < width && y < height)
  {
    localMin = localMax = image[idx];
  }

  sMin[tid] = localMin;
  sMax[tid] = localMax;
  __syncthreads();

  for (int stride = blockDim.x * blockDim.y / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
    {
      sMin[tid] = fminf(sMin[tid], sMin[tid + stride]);
      sMax[tid] = fmaxf(sMax[tid], sMax[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0)
  {
    atomicMinFloat(min, sMin[0]);
    atomicMaxFloat(max, sMax[0]);
  }
}



__global__ void floatToUchar4(float *image, uchar4 *returnImage, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    returnImage[idx].x = image[idx];
    returnImage[idx].y = image[idx];
    returnImage[idx].z = image[idx];
    returnImage[idx].w = 255;
  }
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



__global__ void uchar4ToCufftComplex(cufftComplex* red,cufftComplex* green,cufftComplex* blue, uchar4* ucharImage, int width,int height) {
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x<width && y<height) {
    int idx = width*y+x;
    red[idx] = make_float2(static_cast<float>(ucharImage[idx].x),0.0f);
    green[idx] = make_float2(static_cast<float>(ucharImage[idx].y),0.0f);
    blue[idx] = make_float2(static_cast<float>(ucharImage[idx].z),0.0f);
  }
}

__global__ void ComplexRGBToUchar(cufftComplex* red, cufftComplex* green, cufftComplex* blue, uchar4* returnImage,int width,int height) {
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  if(x<width && y<height) {
    int idx = y * width + x;
    returnImage[idx].x = static_cast<unsigned char>(red[idx].x);
    returnImage[idx].z = static_cast<unsigned char>(green[idx].x);
    returnImage[idx].y = static_cast<unsigned char>(blue[idx].x);
    returnImage[idx].w = 255;
  }
}


__global__ void highpassFilter(cufftComplex* image, int width, int height, float cutoff) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x<width && y < height) {
    int idx = width*y+x;
    int centerX = width/2;
    int centerY = height/2;
    float distance = sqrtf((x-centerX)*(x-centerX) + (y-centerY) * (y-centerY));
    if(distance<cutoff) {
      image[idx].x = 0.0f;
      image[idx].y = 0.0f;
    } 
  }
}

__global__ void NormalizeAndZeroImaginary(cufftComplex* data,int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x<width && y<height) {
    int idx = width*y+x;
    float scale = 1.0f/(width*height);
    data[idx].x*=scale;
    data[idx].y*=scale;

    if(fabs(data[idx].y)<1e-6) {
      data[idx].y= 0.0f;
    }
  }
}

__global__ void fftShift(cufftComplex* data, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x<width/2 && y<height/2) {
    int idx1 = y*width+x;
    int idx2 = (y+height/2)*width+(x+width/2);
    int idx3 = y*width + (x+width/2);
    int idx4 = (y+height/2)*width+x;

    cufftComplex temp1 = data[idx1];
    data[idx1] = data[idx2];
    data[idx2] = temp1;

    cufftComplex temp2 = data[idx3];
    data[idx3] = data[idx4];
    data[idx4] = temp2;
  }
  __syncthreads();
}


void imageFFTImageGenerate(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height) {
  uchar4* d_image;
  rgbCufft complexImage;
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  cufftComplex* red;
  cufftComplex* green;
  cufftComplex* blue;

  cufftComplex* redOut;
  cufftComplex* greenOut;
  cufftComplex* blueOut;

  cudaMalloc(&red,width*height*sizeof(cufftComplex));
  cudaMalloc(&blue,width*height*sizeof(cufftComplex));
  cudaMalloc(&green,width*height*sizeof(cufftComplex));

  cudaMalloc(&redOut,width*height*sizeof(cufftComplex));
  cudaMalloc(&blueOut,width*height*sizeof(cufftComplex));
  cudaMalloc(&greenOut,width*height*sizeof(cufftComplex));

  complexImage.red = red;
  complexImage.blue = blue;
  complexImage.green = green;

  cudaMalloc(&d_image,width*height*sizeof(uchar4));
  cudaMemcpy(d_image,imageLoaded,width*height*sizeof(uchar4),cudaMemcpyHostToDevice);

  uchar4ToCufftComplex<<<blocks,threads>>>(red,blue,green, d_image,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  cufftHandle plan;
  cufftPlan2d(&plan,width,height,CUFFT_C2C);

  cufftExecC2C(plan,red,redOut,CUFFT_FORWARD);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  
  cufftExecC2C(plan,green,greenOut,CUFFT_FORWARD);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  cufftExecC2C(plan,blue,blueOut,CUFFT_FORWARD);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  
  fftShift<<<blocks, threads>>>(redOut, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  fftShift<<<blocks, threads>>>(greenOut, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  fftShift<<<blocks, threads>>>(blueOut, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  float cutoff = 10.0f;
  highpassFilter<<<blocks, threads>>>(redOut, width, height, cutoff);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  highpassFilter<<<blocks, threads>>>(greenOut, width, height, cutoff);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  highpassFilter<<<blocks, threads>>>(blueOut, width, height, cutoff);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  fftShift<<<blocks, threads>>>(redOut, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  fftShift<<<blocks, threads>>>(greenOut, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  fftShift<<<blocks, threads>>>(blueOut, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  cufftExecC2C(plan,redOut,red,CUFFT_INVERSE);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  cufftExecC2C(plan,greenOut,green,CUFFT_INVERSE);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  cufftExecC2C(plan,blueOut,blue,CUFFT_INVERSE);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  NormalizeAndZeroImaginary<<<blocks,threads>>>(red,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  NormalizeAndZeroImaginary<<<blocks,threads>>>(green,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  NormalizeAndZeroImaginary<<<blocks,threads>>>(blue,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  ComplexRGBToUchar<<<blocks,threads>>>(red,green,blue,d_image,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  printf("to complex struct\n");
  cudaMemcpy(returnImage,d_image,width*height*sizeof(uchar4),cudaMemcpyDeviceToHost);

  cufftDestroy(plan);

  cudaFree(d_image);
  checkCuda(cudaFree(complexImage.red));
  checkCuda(cudaFree(complexImage.green));
  checkCuda(cudaFree(complexImage.blue));
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