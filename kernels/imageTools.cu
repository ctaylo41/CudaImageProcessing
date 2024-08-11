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
__global__ void printImage(ComplexRGB *data, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height)
  {
    int idx = width * y + x;
    printf("((%.3f,%.3f),(%.3f,%.3f),(%.3f,%.3f))\n", data[idx].r.real, data[idx].r.imag,data[idx].b.real,data[idx].b.imag, data[idx].g.real,data[idx].g.imag);
  }
}

__global__ void printImage(Complex *data, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x < width && y < height)
  {
    int idx = width * y + x;
    printf("(%.4f,%.4f)\n",data[idx].real,data[idx].imag);
  }
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

/*
__device__ void transpose(Complex* data,int width,int height) {
  __shared__ Complex tile[width][height];

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    int idx = y*width+x;
    tile[y][x] = data[idx];
  }

  __syncthreads();
  int jdx = y * height + x;
  data[jdx] = tile[y][x];

}
*/

__global__ void fft1D(Complex *data, int n, int stride,int width,int height,bool isRow)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(isRow && tid<height || !isRow && tid<width) {
  if (n<=1) return;

  Complex* even;
  Complex* odd;
  cudaMalloc(&even,(n/2)*sizeof(Complex));
  cudaMalloc(&odd,(n/2)*sizeof(Complex));

  for(int i=0;i<n/2;i++) {
    even[i] = data[i*2*stride];
    odd[i] = data[(i*2+1)*stride];
  }

  fft1D<<<1,1>>>(even,n/2,stride,width,height,isRow);
  fft1D<<<1,1>>>(odd,n/2,stride,width,height,isRow);

  for(int k = 0;k<n/2;k++) {
    Complex t = odd[k] *Complex(cos(-2*M_PI*k/n),sin(-2*M_PI*k/n));
    data[k*stride] = even[k] + t;
    data[(k+n/2)*stride] = even[k] - t;
  }

  cudaFree(even);
  cudaFree(odd);
  }
}

__global__ void fft1D(ComplexRGB *data, int width, int height, int step, bool isRow)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  int n = isRow ? width : height;

  if (isRow)
  {
    if (idx < n / 2 && idy < height)
    {
      ComplexRGB even = data[2 * idx + idy * width];
      ComplexRGB odd = data[2 * idx + 1 + idy * width];
      float angle = -2.0f * M_PI * idx / n;
      Complex twid = Complex(cosf(angle), sinf(angle));
      ComplexRGB twiddle(twid, twid, twid);
      ComplexRGB temp = odd * twiddle;
      data[idx + idy * width] = even + temp;
      data[idx + n / 2 + idy * width] = even - temp;
    }
  }
  else
  {
    if (idx < width && idy < n / 2)
    {
      int i = idx * width + idy;
      ComplexRGB even = data[idx + (2 * idy) * width];
      ComplexRGB odd = data[idx + (2 * idy + 1) * width];
      float angle = -2.0f * M_PI * idy / n;
      Complex twid = Complex(cosf(angle), sinf(angle));
      ComplexRGB twiddle(twid, twid, twid);
      ComplexRGB temp = odd * twiddle;
      data[idx + idy * width] = even + temp;
      data[idx + (idy + n / 2) * width] = even - temp;
    }
  }
}

void fftImage(ComplexRGB *data, int width, int height) {
int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);

  dim3 block(16, 16);
  dim3 gridRow((width + block.x - 1) / block.x, height); // Changed this
  dim3 gridCol(width, (height + block.y - 1) / block.y); // Changed this

  for (int step = 1; step < width; step *= 2)
  {
    fft1D<<<gridRow, block>>>(data, width, height, step, true);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }

  printImage<<<blocks,threads>>>(data,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  for (int step = 1; step < height; step *= 2)
  {
    fft1D<<<gridCol, block>>>(data, width, height, step, false);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
}

__global__ void ifft1D(ComplexRGB *data, int width, int height, int step, bool isRow)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  int n = isRow ? width : height;

  if (isRow)
  {
    if (idx < n / 2 && idy < height)
    {
      ComplexRGB even = data[2 * idx + idy * width];
      ComplexRGB odd = data[2 * idx + 1 + idy * width];
      float angle = 2.0f * M_PI * idx / n;
      Complex twid = Complex{cosf(angle), sinf(angle)};
      ComplexRGB twiddle{twid, twid, twid};
      ComplexRGB temp = odd * twiddle;
      data[idx + idy * width] = (even + temp);
      data[idx + n / 2 + idy * width] = (even - temp);
    }
  }
  else
  {
    if (idx < width && idy < n / 2)
    {
      int i = idx * width + idy;
      ComplexRGB even = data[idx + (2 * idy) * width];
      ComplexRGB odd = data[idx + (2 * idy + 1) * width];
      float angle = 2.0f * M_PI * idy / n;
      Complex twid = Complex{cosf(angle), sinf(angle)};
      ComplexRGB twiddle{twid, twid, twid};
      ComplexRGB temp = odd * twiddle;
      data[idx + idy * width] = (even + temp);
      data[idx + (idy + n / 2) * width] = (even - temp);
    }
  }
}

__global__ void ifft1D(Complex *data, int width, int height, int step, bool isRow)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;

  int n = isRow ? width : height;

  if (isRow)
  {
    if (idx < n / 2 && idy < height)
    {
      Complex even = data[2 * idx + idy * width];
      Complex odd = data[2 * idx + 1 + idy * width];
      float angle = 2.0f * M_PI * idx / n;
      Complex twiddle(cosf(angle), sinf(angle));
      Complex temp = odd * twiddle;
      data[idx + idy * width] = (even + temp) / n;
      data[idx + n / 2 + idy * width] = (even - temp) / n;
    }
  }
  else
  {
    if (idx < width && idy < n / 2)
    {
      int i = idx * width + idy;
      Complex even = data[idx + (2 * idy) * width];
      Complex odd = data[idx + (2 * idy + 1) * width];
      float angle = 2.0f * M_PI * idy / n;
      Complex twiddle(cosf(angle), sinf(angle));
      Complex temp = odd * twiddle;
      data[idx + idy * width] = (even + temp) / n;
      data[idx + (idy + n / 2) * width] = (even - temp) / n;
    }
  }
}

void fftImage(Complex *data, int width, int height)
{
  int threadsPerBlock = 1024; // Maximum number of threads per block
  int numBlocks = (height + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks
  for (int i = 0; i<height;i++)
  {
    fft1D<<<numBlocks,threadsPerBlock>>>(&data[i*width],width,1,width,height,true);
  }

  /*
  Complex* temp = new Complex[height];
  for(int i = 0; i < width; i++) {
  for(int j = 0; j < height; j++) {
    temp[j] = data[j*width + i];
  }
  fft1D(temp, height, 1);
  for(int j = 0; j < height; j++) {
    data[j*width + i] = temp[j];
    }
  }
  */
  for(int i=0;i<width*height;i++) {
    printf("(%.4f,%.4f)\n",data[i].real,data[i].imag);
  }
  //delete[] temp;
}



void ifftImage(Complex *data, int width, int height)
{
  dim3 block(16, 16);
  dim3 gridRow((width + block.x - 1) / block.x, height); // Changed this
  dim3 gridCol(width, (height + block.y - 1) / block.y); // Changed this

  for (int step = 1; step < width; step *= 2)
  {
    ifft1D<<<gridRow, block>>>(data, width, height, step, true);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  printf("rows done\n");
  for (int step = 1; step < height; step *= 2)
  {
    ifft1D<<<gridCol, block>>>(data, width, height, step, false);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
}

__global__ void normalizeComplex(ComplexRGB *data, float scalar, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    data[idx].r = data[idx].r * scalar;
    data[idx].g = data[idx].g * scalar;
    data[idx].b = data[idx].b * scalar;
  }
}

void ifftImage(ComplexRGB *data, int width, int height)
{
  dim3 block(16, 16);
  dim3 gridRow((width + block.x - 1) / block.x, height); // Changed this
  dim3 gridCol(width, (height + block.y - 1) / block.y); // Changed this

  for (int step = 1; step < width; step *= 2)
  {
    ifft1D<<<gridRow, block>>>(data, width, height, step, true);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  printf("rows done\n");
  for (int step = 1; step < height; step *= 2)
  {
    ifft1D<<<gridCol, block>>>(data, width, height, step, false);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);

  float scalar = 1.0f / (width * height);
  normalizeComplex<<<blocks, threads>>>(data, scalar, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
}

__global__ void grayScaleToComplex(uchar4 *imageGrayScale, Complex *imageComplex, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height)
  {
    Complex num = Complex{static_cast<float>(imageGrayScale[x * width + y].x), static_cast<float>(0)};
    imageComplex[y * width + x] = num;
  }
}

__global__ void computeMagnitude(Complex *complexImage, float *magnitudeImage, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    magnitudeImage[idx] = complexImage[idx].magnitude();
  }
}

__global__ void logImage(float *image, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    float c = 255.0f / logf(1.0f + 255);
    image[idx] = c * logf(1.0f + fabsf(image[idx]));
  }
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

__global__ void normalize(float *image, float *min, float *max, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height)
  {
    int idx = y * width + x;
    float normalizedValue = (image[idx] - *min) / (*max - *min) * 255.0f;
    image[idx] = normalizedValue;
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

__global__ void fftShift(Complex *input, Complex *output, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int newX = (x + width / 2) % width;
    int newY = (y + height / 2) % height;

    int oldIndex = y * width + x;
    int newIndex = newY * width + newX;

    output[newIndex] = input[oldIndex];
  }
}

__global__ void complexToFloat(Complex *input, float *output, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    output[idx] = input[idx].real;
  }
}

void putBackTogether(uchar4 *output, uchar4 *input, int width, int height)
{
  ComplexRGB *d_image;

  checkCuda(cudaMallocManaged(&d_image, width * height * sizeof(ComplexRGB)));
}

void imageFFTImageGenerate(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height)
{
  uchar4 *d_returnImage;
  uchar4 *d_imageLoaded;
  Complex *complexImage;
  Complex *shiftedComplexImage;
  float *d_min, *d_max;
  float h_min, h_max;
  float *mangnitudeImage;
  int imgSize = width * height;
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  checkCuda(cudaMallocManaged(&d_returnImage, imgSize * sizeof(uchar4)));
  checkCuda(cudaMallocManaged(&d_imageLoaded, imgSize * sizeof(uchar4)));
  checkCuda(cudaMallocManaged(&shiftedComplexImage, imgSize * sizeof(Complex)));
  checkCuda(cudaMallocManaged(&complexImage, imgSize * sizeof(Complex)));
  checkCuda(cudaMallocManaged(&mangnitudeImage, width * height * sizeof(float)));
  checkCuda(cudaMemcpy(d_imageLoaded, imageLoaded, imgSize * sizeof(uchar4), cudaMemcpyHostToDevice));
  imageGrayScale<<<blocks, threads>>>(d_returnImage, d_imageLoaded, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  printf("grayscale done\n");
  grayScaleToComplex<<<blocks, threads>>>(d_returnImage, complexImage, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  printf("to complex done\n");
  fftImage(complexImage, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  fftShift<<<blocks, threads>>>(complexImage, shiftedComplexImage, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  printf("fft done columns\n");
  computeMagnitude<<<blocks, threads>>>(shiftedComplexImage, mangnitudeImage, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  printf("dont columns\n");
  logImage<<<blocks, threads>>>(mangnitudeImage, width, height);
  printf("logged entries\n");
  cudaMalloc(&d_min, sizeof(float));
  cudaMalloc(&d_max, sizeof(float));

  findMinMax<<<blocks, threads, 2 * threads.x * threads.y * sizeof(float)>>>(mangnitudeImage, width, height, d_min, d_max);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  normalize<<<blocks, threads>>>(mangnitudeImage, d_min, d_max, width, height);
  printf("normaliz\n");
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  floatToUchar4<<<blocks, threads>>>(mangnitudeImage, d_returnImage, width, height);
  printf("float to char\n");

  cudaMemcpy(returnImage, d_returnImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

  cudaFree(d_returnImage);
  cudaFree(d_imageLoaded);
  cudaFree(complexImage);
  cudaFree(shiftedComplexImage);
  cudaFree(d_min);
  cudaFree(d_max);
  printf("memory free\n");
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

__global__ void uchar4ToComplexRGB(uchar4 *image, ComplexRGB *complexImage, int width, int height)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height)
  {
    int idx = y * width + x;
    complexImage[idx] = ComplexRGB{Complex{static_cast<float>(image[idx].x), 0}, Complex{static_cast<float>(image[idx].y), 0}, Complex{static_cast<float>(image[idx].z), 0}};
  }
}
__global__ void applyLowPassFilter(Complex *data, int width, int height, float cutoff)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    float dist = sqrtf((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2));
    if (dist > cutoff)
    {
      data[idx].real = 0.0f;
      data[idx].imag = 0.0f;
    }
  }
}

__global__ void applyLowPassFilter(ComplexRGB *data, int width, int height, float cutoff)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;

    float dist = sqrtf((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2));
    if (dist < cutoff)
    {
      data[idx].r.real = 0.0f;
      data[idx].r.imag = 0.0f;
      data[idx].g.real = 0.0f;
      data[idx].g.imag = 0.0f;
      data[idx].b.real = 0.0f;
      data[idx].b.imag = 0.0f;
    }
  }
}

__global__ void complexRGBToUchar4(ComplexRGB *input, uchar4 *output, int width, int height)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    output[idx].x = static_cast<unsigned char>(input[idx].r.real);
    output[idx].y = static_cast<unsigned char>(input[idx].g.real);
    output[idx].z = static_cast<unsigned char>(input[idx].b.real);
    output[idx].w = 255;
  }
}


void __device__ swap(Complex& a, Complex& b) {
  Complex temp = a;
  a = b;
  b = temp;
}



__global__ void bitReversalKernel(Complex* data, int rows,int cols) {
  int row = blockIdx.y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(row < rows && i < cols) {
    int j = 0;
    int bits = 0;
    int temp = cols;
    while (temp > 1) {
      temp >>=1;
      bits++;
    }

    for(int k = 0;k<bits;k++) {
      j = (j << 1) | ((i >> k) & 1);
    }
    if(i<j) {
      swap(data[row*cols+i],data[row*cols+j]);
    }
  }
}

__global__ void fftKernel(Complex* data, int width,int height) {
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if(row<height) {
    printf("%d\n",row);
    Complex* rowData = data+row*width;
    for(int len=2;len<=width;len<<=1) {
      float angle = -2 * M_PI/len;
      Complex wlen = {cosf(angle),sinf(angle)};
      for(int i=0;i<width;i+=len) {
        Complex w = {1,0};
        for(int j=0;j<len/2;j++) {
          Complex u = rowData[i+j];
          Complex v = rowData[i+j+len/2]*w;
          rowData[i+j] = u+v;
          rowData[i+j+len/2] = u-v;
          w = w*wlen;
        }
      }
    }
  }
}

void compressImage(uchar4 *outputImage, uchar4 *inputImage, int width, int height)
{
  ComplexRGB *complexImage;
  uchar4 *d_image;
  uchar4 *compressedImage;
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  checkCuda(cudaMalloc(&complexImage, width * height * sizeof(ComplexRGB)));
  checkCuda(cudaMalloc(&compressedImage, width * height * sizeof(uchar4)));
  checkCuda(cudaMalloc(&d_image, width * height * sizeof(uchar4)));
  checkCuda(cudaMemcpy(d_image, inputImage, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));
  uchar4ToComplexRGB<<<blocks, threads>>>(d_image, complexImage, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  

  fftImage(complexImage, width, height);
  printImage<<<blocks,threads>>>(complexImage,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  
  
  float max_distance = sqrtf((width / 2) * (width / 2) + (height / 2) * (height / 2));
  float cutoff = 0.15f * max_distance; // 15% of the maximum distance
  ifftImage(complexImage, width, height);
  complexRGBToUchar4<<<blocks,threads>>>(complexImage,compressedImage,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  checkCuda(cudaMemcpy(outputImage, compressedImage, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
}

bool test()
{
  std::srand(std::time(nullptr));
  int width = 4;
  int height = 2;
  Complex *image = (Complex*)malloc(width*height*sizeof(Complex));   
  Complex *d_image;
  checkCuda(cudaMalloc(&d_image,width*height*sizeof(Complex)));
  image[0] = Complex{4,0};
  image[1] = Complex{9,0};
  image[2] = Complex{13,0};
  image[3] = Complex{5,0};
  image[4] = Complex{2,0};
  image[5] = Complex{19,0};
  image[6] = Complex{1,0};
  image[7] = Complex{29,0};
  checkCuda(cudaMemcpy(d_image,image,width*height*sizeof(Complex),cudaMemcpyHostToDevice));
  dim3 blockSize(256,1);
  dim3 gridSize((width+blockSize.x-1)/blockSize.x,height); 
  bitReversalKernel<<<gridSize,blockSize>>>(d_image,height,width);
  checkCuda(cudaGetLastError());
  cudaDeviceSynchronize();
  printImage<<<blockSize,gridSize>>>(d_image,width,height);
  checkCuda(cudaGetLastError());
  cudaDeviceSynchronize();
  fftKernel<<<blockSize.x,gridSize.x>>>(d_image,width,height);
  checkCuda(cudaGetLastError());
  cudaDeviceSynchronize();
  Complex* r_image = (Complex*)malloc(width*height*sizeof(Complex));
  printf("bruh\n");

  checkCuda(cudaMemcpy(r_image,d_image,width*height*sizeof(Complex),cudaMemcpyDeviceToHost));

  for(int i=0;i<width*height;i++) {
    printf("(%.4f,%.4f)\n",r_image[i].real,r_image[i].imag);
  }
  cudaFree(d_image);
  free(image);
  return true;
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