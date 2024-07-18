#ifndef IMAGELOAD_H
#define IMAGELOAD_H

__global__ void imageLoad(unsigned char* image, uchar4* imageLoaded, size_t imgSize);
void imageLoadWrapper(unsigned char* image, uchar4* imageLoaded, size_t imgSize);
#endif // IMAGELOAD_H