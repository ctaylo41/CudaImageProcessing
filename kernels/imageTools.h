#ifndef IMAGETOOLS_H
#define IMAGETOOLS_H
#include <vector>
__global__ void imageLoad(unsigned char* image, uchar4* imageLoaded, size_t imgSize);
__global__ void imageWrite(unsigned char* image, uchar4* pixels, int width, int height);
__global__ void imageGrayScale(uchar4* returnImage, uchar4* imageLoaded, int width, int height);
__global__ void imageSobelEdge(uchar4* returnImage, uchar4* imageLoaded, int width, int height);
__global__ void imageGaussianBlur(uchar4* returnImage, uchar4* imageLoaded, int width, int height, int kernalSize, float* kernal);
__global__ void imageMeanBlur(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height, int kernalSize);
void imageLoadWrapper(unsigned char* image, uchar4* imageLoaded, size_t imgSize);
void imageWriteWrapper(const char* filename, uchar4* pixels,int width,int height);
void imageGrayScaleWrapper(uchar4* returnImage, uchar4* imageLoaded, int width,int height);
void imageSobelEdgeWrapper(uchar4* returnImage, uchar4* imageLoaded, int width, int height);
void imageGaussianBlurWrapper(uchar4* returnImage, uchar4* imageLoaded, int width, int height, int kernalSize, float sigma);
void imageMeanBlurWrapper(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height);
void imageFFTImageGenerate(uchar4 *returnImage, uchar4 *imageLoaded, int width, int height);
float* generateGaussianKernal(int kernalSize, float sigma);
bool test();
#endif // IMAGETOOLS_H