/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "simpleLinearBlurFilter.hpp"

/*
 * CUDA Kernel Device code
 *
 */
__global__ void applySimpleLinearBlurFilter(uchar *r, uchar *g, uchar *b)
{
    // Not using the 2d grid + shared memory approach

    /* 
    Using the CPU version as reference... as both versions are to have min mean diff

    for(int y = 0; y < rows; ++y)
    {
        for(int x = 1; x < columns-1; ++x)
        {
            Vec3b rgb0 = img.at<Vec3b>(y, x-1);
            Vec3b rgb1 = img.at<Vec3b>(y, x);
            Vec3b rgb2 = img.at<Vec3b>(y, x+1);
            b[y*rows+x] = (rgb0[0] + rgb1[0] + rgb2[0])/3;
            g[y*rows+x] = (rgb0[1] + rgb1[1] + rgb2[1])/3;
            r[y*rows+x] = (rgb0[2] + rgb1[2] + rgb2[2])/3;
        }
    }
    */

    int threadId = blockDim.x * blockIdx.x + threadIdx.x; // because we have a 1d grid and 1d block for 2d - serialised img
    int max_pixels = d_rows * d_columns;
    if(threadId < max_pixels) // total_threads = d_rows * d_columns
    {
        // Not using the 2d grid + shared memory approach

        // Apply a simple filter that averages the RGB values to the left and right of the pixel at the current thread id location
        // Another area for improvement is handling when the current thread is at the let or right edge of the imput image
        /*
            Edge case: as our blur 1d kernel takes the i-1 , i and i+1 for ith pixel, we cant run it for the first and last column. 
            So as per the CPU version, we ignore them. 
        */
        int row = threadId / d_columns;
        int col = threadId % d_columns;

        if(col == 0 || col == d_columns-1) return;

        // for all remaining indices, just take avg 
        r[threadId] = (r[threadId-1]+r[threadId]+r[threadId+1]) / 3;
        g[threadId] = (g[threadId-1]+g[threadId]+g[threadId+1]) / 3;
        b[threadId] = (b[threadId-1]+b[threadId]+b[threadId+1]) / 3;
    }

}

__host__ float compareColorImages(uchar *r0, uchar *g0, uchar *b0, uchar *r1, uchar *g1, uchar *b1, int rows, int columns)
{
    cout << "Comparing actual and test pixel arrays\n";
    int numImagePixels = rows * columns;
    int imagePixelDifference = 0.0;

    for(int r = 0; r < rows; ++r)
    {
        for(int c = 0; c < columns; ++c)
        {
            uchar image0R = r0[r*rows+c];
            uchar image0G = g0[r*rows+c];
            uchar image0B = b0[r*rows+c];
            uchar image1R = r1[r*rows+c];
            uchar image1G = g1[r*rows+c];
            uchar image1B = b1[r*rows+c];
            imagePixelDifference += ((abs(image0R - image1R) + abs(image0G - image1G) + abs(image0B - image1B))/3);
        }
    }

    float meanImagePixelDifference = imagePixelDifference / numImagePixels;
    float scaledMeanDifferencePercentage = (meanImagePixelDifference / 255);
    printf("meanImagePixelDifference: %f scaledMeanDifferencePercentage: %f\n", meanImagePixelDifference, scaledMeanDifferencePercentage);
    return scaledMeanDifferencePercentage;
}

__host__ void allocateDeviceMemory(int rows, int columns)
{

    //Allocate device constant symbols for rows and columns
    cudaMemcpyToSymbol(d_rows, &rows, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_columns, &columns, sizeof(int), 0, cudaMemcpyHostToDevice);
    int max_pixels = rows * columns;
    cudaMemcpyToSymbol(d_max_pixels, &max_pixels, sizeof(int), 0, cudaMemcpyHostToDevice); // Had to add this as the size of shared variable in kerel can only be specified using a constant and not pdt of 2 constants
}

__host__ void executeKernel(uchar *r, uchar *g, uchar *b, int rows, int columns, int threadsPerBlock)
{
    cout << "Executing kernel\n";
    //Launch the convert CUDA Kernel
    int blocksPerGrid = (rows * columns) / threadsPerBlock;
 
    applySimpleLinearBlurFilter<<<blocksPerGrid, threadsPerBlock>>>(r, g, b); // 1d kernel
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    std::string inputImage = "sloth.png";
    std::string outputImage = "grey-sloth.png";
    std::string currentPartId = "test";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " currentPartId: " << currentPartId << " threadsPerBlock: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, currentPartId, threadsPerBlock};
}

__host__ std::tuple<int, int, uchar *, uchar *, uchar *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);
    
    const int rows = img.rows;
    const int columns = img.cols;
    size_t size = sizeof(uchar) * rows * columns;

    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    uchar *r, *g, *b;
    cudaMallocManaged(&r, size);
    cudaMallocManaged(&g, size);
    cudaMallocManaged(&b, size);
    
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < columns; ++x)
        {
            Vec3b rgb = img.at<Vec3b>(y, x);
            r[y*rows+x] = rgb.val[0];
            g[y*rows+x]= rgb.val[1];
            b[y*rows+x] = rgb.val[2];
        }
    }

    return {rows, columns, r, g, b};
}

__host__ std::tuple<uchar *, uchar *, uchar *>applyBlurKernel(std::string inputImage)
{
    cout << "CPU applying kernel\n";
    Mat img = imread(inputImage, IMREAD_COLOR);
    const int rows = img.rows;
    const int columns = img.cols;

    uchar *r = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *g = (uchar *)malloc(sizeof(uchar) * rows * columns);
    uchar *b = (uchar *)malloc(sizeof(uchar) * rows * columns);

    for(int y = 0; y < rows; ++y)
    {
        for(int x = 1; x < columns-1; ++x)
        {
            Vec3b rgb0 = img.at<Vec3b>(y, x-1);
            Vec3b rgb1 = img.at<Vec3b>(y, x);
            Vec3b rgb2 = img.at<Vec3b>(y, x+1);
            b[y*rows+x] = (rgb0[0] + rgb1[0] + rgb2[0])/3;
            g[y*rows+x] = (rgb0[1] + rgb1[1] + rgb2[1])/3;
            r[y*rows+x] = (rgb0[2] + rgb1[2] + rgb2[2])/3;
        }
    }

    return {r, g, b};
}

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    std::string currentPartId = get<2>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<3>(parsedCommandLineArgsTuple);
    try 
    {
        auto[rows, columns, r, g, b] = readImageFromFile(inputImage);

        executeKernel(r, g, b, rows, columns, threadsPerBlock);

        Mat colorImage(rows, columns, CV_8UC3);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < columns; ++x)
            {
                colorImage.at<Vec3b>(y,x) = Vec3b(b[y*rows+x], g[y*rows+x], r[y*rows+x]);
            }
        }

        imwrite(outputImage, colorImage, compression_params);

        auto[test_r, test_g, test_b] = applyBlurKernel(inputImage);
        
        float scaledMeanDifferencePercentage = compareColorImages(r, g, b, test_r, test_g, test_b, rows, columns) * 100;
        cout << "Mean difference percentage: " << scaledMeanDifferencePercentage << "\n";

        cleanUpDevice();
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}