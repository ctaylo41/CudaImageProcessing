#ifndef IMAGETOOLS_H
#define IMAGETOOLS_H

__global__ void imageLoad(unsigned char* image, uchar4* imageLoaded, size_t imgSize);
__global__ void imageWrite(unsigned char* image, uchar4* pixels, int width, int height);
__global__ void imageGrayScale(uchar4* returnImage, uchar4* imageLoaded, size_t imgSize);
__global__ void imageSobelEdge(uchar4* returnImage, uchar4* imageLoaded, int width, int height);
void imageLoadWrapper(unsigned char* image, uchar4* imageLoaded, size_t imgSize);
void imageWriteWrapper(const char* filename, uchar4* pixels,int width,int height);
void imageGrayScaleWrapper(uchar4* returnImage, uchar4* imageLoaded, size_t imgSize);
void imageSobelEdgeWrapper(uchar4* returnImage, uchar4* imageLoaded, int width, int height);
#endif // IMAGETOOLS_H