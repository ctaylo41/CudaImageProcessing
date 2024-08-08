#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <assert.h>
#include "imageTools.h"
#include "stb_image_write.h"
#include <vector>
#include <cfloat>
#include <float.h>

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

  __device__ __host__ ComplexRGB operator+(const ComplexRGB &Other)
  {
    return ComplexRGB{r + Other.r, g + Other.g, b + Other.b};
  }

  __device__ __host__ ComplexRGB operator-(const ComplexRGB &Other)
  {
    return ComplexRGB{r - Other.r, g - Other.g, b - Other.b};
  }

  __device__ __host__ ComplexRGB operator*(const ComplexRGB &Other)
  {
    return ComplexRGB{r * Other.r, g * Other.g, b * Other.b};
  }

  __device__ __host__ float3 magnitude() const
  {
    return float3{r.magnitude(), g.magnitude(), b.magnitude()};
  }

  __device__ __host__ ComplexRGB operator/(float scalar) {
    return ComplexRGB(r/scalar,g/scalar,b/scalar);
  }
};

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

__global__ void fft1D(Complex *data, int width, int height, int step, bool isRow)
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
      float angle = -2.0f * M_PI * idx / n;
      Complex twiddle(cosf(angle), sinf(angle));
      Complex temp = odd * twiddle;
      data[idx + idy * width] = even + temp;
      data[idx + n / 2 + idy * width] = even - temp;
    }
  }
  else
  {
    if (idx < width && idy < n / 2)
    {
      int i = idx * width + idy;
      Complex even = data[idx + (2 * idy) * width];
      Complex odd = data[idx + (2 * idy + 1) * width];
      float angle = -2.0f * M_PI * idy / n;
      Complex twiddle(cosf(angle), sinf(angle));
      Complex temp = odd * twiddle;
      data[idx + idy * width] = even + temp;
      data[idx + (idy + n / 2) * width] = even - temp;
    }
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
      ComplexRGB twiddle(twid,twid,twid);
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
      ComplexRGB twiddle(twid,twid,twid);
      ComplexRGB temp = odd * twiddle;
      data[idx + idy * width] = even + temp;
      data[idx + (idy + n / 2) * width] = even - temp;
    }
  }
}

void fftImage(ComplexRGB *data, int width, int height)
{
  dim3 block(16, 16);
  dim3 gridRow((width + block.x - 1) / block.x, height); // Changed this
  dim3 gridCol(width, (height + block.y - 1) / block.y); // Changed this

  for (int step = 1; step < width; step *= 2)
  {
    fft1D<<<gridRow, block>>>(data, width, height, step, true);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  printf("rows done\n");
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
      Complex twid = Complex(cosf(angle), sinf(angle));
      ComplexRGB twiddle(twid,twid,twid);
      ComplexRGB temp = odd * twiddle;
      data[idx + idy * width] = (even + temp) / n;
      data[idx + n / 2 + idy * width] = (even - temp) / n;
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
      Complex twid = Complex(cosf(angle), sinf(angle));
      ComplexRGB twiddle(twid, twid, twid);
      ComplexRGB temp = odd * twiddle;
      data[idx + idy * width] = (even + temp) / n;
      data[idx + (idy + n / 2) * width] = (even - temp) / n;
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
  dim3 block(16, 16);
  dim3 gridRow((width + block.x - 1) / block.x, height); // Changed this
  dim3 gridCol(width, (height + block.y - 1) / block.y); // Changed this

  for (int step = 1; step < width; step *= 2)
  {
    fft1D<<<gridRow, block>>>(data, width, height, step, true);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  printf("rows done\n");
  for (int step = 1; step < height; step *= 2)
  {
    fft1D<<<gridCol, block>>>(data, width, height, step, false);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
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

void putBackTogether(uchar4* output, uchar4* input, int width, int height) {
  ComplexRGB* d_image;
  
  checkCuda(cudaMallocManaged(&d_image,width*height*sizeof(ComplexRGB)));


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

__global__ void uchar4ToComplexRGB(uchar4* image, ComplexRGB* complexImage, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x<width && y<height) {
    int idx = y * width + x;
    complexImage[idx].r = image[idx].x;
    complexImage[idx].g = image[idx].y;
    complexImage[idx].b = image[idx].z;
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

__global__ void applyLowPassFilter(ComplexRGB* data, int width, int height, float cutoff)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height)
  {
    int idx = y * width + x;
    float dist = sqrtf((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2));
    if(dist > cutoff) {
      data[idx].r.real = 0.0f;
      data[idx].r.imag = 0.0f;
      data[idx].g.real = 0.0f;
      data[idx].g.imag = 0.0f;
      data[idx].b.real = 0.0f;
      data[idx].b.imag = 0.0f;
    }
  }
}

__global__ void complexRGBToUchar4(ComplexRGB* input, uchar4* output, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if(x < width && y < height) {
    int idx = y * width + x;
    output[idx].x = static_cast<unsigned char>(input[idx].r.real);
    output[idx].y = static_cast<unsigned char>(input[idx].g.real);
    output[idx].z = static_cast<unsigned char>(input[idx].b.real);
    output[idx].w = 255;
  }
}

void compressImage(uchar4* outputImage,uchar4* inputImage, int width, int height) {
  ComplexRGB* complexImage;
  uchar4* d_image;
  uchar4* compressedImage;
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);
  checkCuda(cudaMalloc(&complexImage,width*height*sizeof(ComplexRGB)));
  checkCuda(cudaMalloc(&compressedImage,width*height*sizeof(uchar4)));
  checkCuda(cudaMalloc(&d_image,width*height*sizeof(uchar4)));
  checkCuda(cudaMemcpy(d_image,inputImage,width*height*sizeof(uchar4),cudaMemcpyHostToDevice));
  uchar4ToComplexRGB<<<blocks,threads>>>(d_image,complexImage,width,height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  fftImage(complexImage,width,height);
  float max_distance = sqrtf((width / 2) * (width / 2) + (height / 2) * (height / 2));
  float cutoff = 0.15f * max_distance; // 15% of the maximum distance

  applyLowPassFilter<<<blocks,threads>>>(complexImage,width,height,cutoff);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());

  ifftImage(complexImage,width,height);

  complexRGBToUchar4<<<blocks,threads>>>(complexImage,compressedImage,width,height);
  checkCuda(cudaMemcpy(outputImage,compressedImage,width*height*sizeof(uchar4),cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
}

bool test()
{
  int width = 2;
  int height = 2;
  Complex *image = (Complex *)malloc(width * height * sizeof(Complex));
  image[0] = Complex{85, 0};
  image[1] = Complex{175, 0};
  image[2] = Complex{24, 0};
  image[3] = Complex{98, 0};
  Complex *image_device;
  Complex *shifted;
  checkCuda(cudaMallocManaged(&image_device, width * height * sizeof(Complex)));
  checkCuda(cudaMallocManaged(&shifted, width * height * sizeof(Complex)));

  checkCuda(cudaMemcpy(image_device, image, width * height * sizeof(Complex), cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(shifted, image, width * height * sizeof(Complex), cudaMemcpyHostToDevice));
  dim3 block(16, 16);
  dim3 gridRow((width + block.x - 1) / block.x, height); // Changed this
  dim3 gridCol(width, (height + block.y - 1) / block.y); // Changed this

  for (int step = 1; step < width; step *= 2)
  {
    fft1D<<<gridRow, block>>>(image_device, width, height, step, true);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  for (int step = 1; step < width; step *= 2)
  {
    fft1D<<<gridRow, block>>>(image_device, width, height, step, false);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  /*
  int threadsPerBlock = 16;
  int numBlocksX = (width + threadsPerBlock - 1) / threadsPerBlock;
  int numBlocksY = (height + threadsPerBlock - 1) / threadsPerBlock;
  dim3 blocks(numBlocksX, numBlocksY);
  dim3 threads(threadsPerBlock, threadsPerBlock);

  fftShift<<<blocks,threads>>>(image_device, shifted, width, height);
  cudaDeviceSynchronize();
  checkCuda(cudaGetLastError());
  */
  for (int step = 1; step < width; step *= 2)
  {
    ifft1D<<<gridRow, block>>>(image_device, width, height, step, true);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }
  for (int step = 1; step < width; step *= 2)
  {
    ifft1D<<<gridRow, block>>>(image_device, width, height, step, false);
    cudaDeviceSynchronize();
    checkCuda(cudaGetLastError());
  }

  Complex *host_image = (Complex *)malloc(width * height * sizeof(Complex));

  checkCuda(cudaMemcpy(host_image, shifted, width * height * sizeof(Complex), cudaMemcpyDeviceToHost));
  for (int i = 0; i < width * height; i++)
  {
    printf("(%.4f,%.4f)\n", host_image[i].real, host_image[i].imag);
  }
  free(host_image);
  cudaFree(image);
  cudaFree(shifted);
  cudaFree(image_device);
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