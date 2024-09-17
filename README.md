# CUDA Image Processing

This project demonstrates image processing using CUDA. It includes functions for performing Gaussian Blur, Median Blur, Sobel Edge Detection and Grayscaling.
Currently the FFT is being added for image compression.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Building the Project](#building-the-project)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Key Components](#key-components)

## Introduction

The CUDA Image Processing project leverages the power of CUDA to perform efficient image processing tasks. It includes functions for converting images to complex format, performing FFT and inverse FFT, and normalizing the results.

## Requirements

- CUDA Toolkit
- CMake
- GCC
- NVIDIA GPU with CUDA support

## Building the Project

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd CudaImageProcessing
    ```

2. Create a build directory and navigate to it:
    ```sh
    mkdir build
    cd build
    ```

3. Configure the project using CMake:
    ```sh
    cmake ..
    ```

4. Build the project:
    ```sh
    make -j10
    ```