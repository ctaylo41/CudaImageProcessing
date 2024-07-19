#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <cstdlib> 
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#include "kernels/imageLoad.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"



bool saveImage(const char* filename, uchar4* pixels, int width, int height) {
    // Assuming uchar4 is a struct of 4 unsigned chars (RGBA)
    // Convert uchar4* to unsigned char* required by stbi_write_png
    std::cout << "inside save image" << std::endl;
    unsigned char* data = new unsigned char[width * height * 4];    
    for (int i = 0; i < width * height; ++i) {
        std::cout << "R: " << pixels[i].x << std::endl;

        data[i * 4 + 0] = pixels[i].x; // R
        data[i * 4 + 1] = pixels[i].y; // G

        data[i * 4 + 2] = pixels[i].z; // B
        data[i * 4 + 3] = pixels[i].w; // A
    }

    // Write to file
    int result = stbi_write_jpg(filename, width, height, 4, data, width * 4);

    delete[] data; // Clean up the temporary array

    return result != 0; // stbi_write_png returns 0 on failure, non-0 on success
}

uchar4* loadImage(const char* filename,int* width, int *height) {
    int channels;
    unsigned char* img = stbi_load(filename,width,height,&channels,STBI_rgb_alpha);
    if (img == NULL) {
        std::cerr << "Error Loading Image: " << filename << std::endl;
        exit(1);
    }
    
    size_t imgSize = (*width) * (*height);

    uchar4* d_image;
    imageLoadWrapper(img,d_image,imgSize);
    stbi_image_free(img);
    return d_image;
}

int main(int, char**){
    int width; 
    int height;
    uchar4* img = loadImage("/home/paperspace/CudaImageProcessing/image/thumb.gif",&width,&height);
    
    saveImage("/home/paperspace/CudaImageProcessing/image/deerNew.png",img,width,height);
    return 0;
}
