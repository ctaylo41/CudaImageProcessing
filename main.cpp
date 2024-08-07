#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstdlib> 
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#include "kernels/imageTools.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

uchar4* loadImage(const char* filename,int* width, int *height) {
    int channels;
    unsigned char* img = stbi_load(filename,width,height,&channels,STBI_rgb_alpha);
    if (img == NULL) {
        std::cerr << "Error Loading Image: " << filename << std::endl;
        exit(1);
    }
    
    size_t imgSize = (*width) * (*height);
    uchar4* d_image = new uchar4[imgSize];
    imageLoadWrapper(img,d_image,imgSize);
    
    stbi_image_free(img);
    return d_image;
}

int main(int, char**){
    int width; 
    int height;

    uchar4* img = loadImage("/home/paperspace/CudaImageProcessing/image/thumb.gif",&width,&height);
    std::cout << "\nload image\n" << std::endl;

    //imageGrayScaleWrapper(img,img,width,height);
    //imageSobelEdgeWrapper(img,img,width,height);
    //imageWriteWrapper("/home/paperspace/CudaImageProcessing/image/output.jpg",img,width,height);
    //imageGaussianBlurWrapper(img,img,width,height,3,7.0);
    //imageMeanBlurWrapper(img,img,width,height);
    imageFFTImageGenerate(img,img,width,height);
    imageWriteWrapper("/home/paperspace/CudaImageProcessing/image/output.jpg",img,width,height);
    delete[] img;
    return 0;
}
