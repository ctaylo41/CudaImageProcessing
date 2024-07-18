#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#include "kernels/imageLoad.h"



uchar4* loadImage(const char* filename,int* width, int *height) {
    int channels;
    unsigned char* img = stbi_load(filename,width,height,&channels,STBI_rgb_alpha);
    if (img = NULL) {
        std::cerr << "Error Loading Image: " << filename << std::endl;
        exit(1);
    }
    
    size_t imgSize = (*width) * (*height);

    uchar4* d_image;
    imageLoadWrapper(img,d_image,imgSize);
    return d_image;
}


int main(int, char**){
    int* width;
    int* height;
    
    uchar4* img = loadImage("image/deer.jpg",width,height);
    std::cout << "width = " << *width << " height = " << *height << std::endl;
    return 0;
}
