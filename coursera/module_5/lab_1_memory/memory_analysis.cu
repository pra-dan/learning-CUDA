#include <iostream>
#include <fstream>
#include <string>
#include <tuple>
#include <ctime>
#include <cstdlib>

// Constant memory declaration
__constant__ int constant_search_value;
__constant__ int d_numElements;
__constant__ int d_threadSpan;
__constant__ int d_numThreads;

// Global Memory Kernel: Accesses d_input directly from global memory
__global__ void globalMemoryKernel(int* d_input, int d_numThreads)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < d_numThreads)
    {
        // Perform a simple operation: sum the input value 1000 times
        int sum = 0;
        for (int i = 0; i < 1000; i++)
        {
            sum += d_input[threadId];
        }
        d_input[threadId] = sum; // Write back to global memory
    }
}

// Shared Memory Kernel: Copies data to shared memory, then uses it
/*
    This actually doesn't share the array 'sharedInput' among threads.
    Rather, its just to show one non-useful way to access the shared memory as here,
    each thread uses its own local copy. True shared usage would look like:

    // Each thread reads from shared memory values written by multiple threads
    sum += sharedInput[threadIdx.x - 1] + sharedInput[threadIdx.x] + sharedInput[threadIdx.x + 1];
    i.e., thread.x is using the value from prev and next thread.
*/
__global__ void sharedMemoryKernel(int* d_input, int d_numThreads)
{
    extern __shared__ int sharedInput[];
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int localId = threadIdx.x;

    // Copy data from global to shared memory
    if (threadId < d_numThreads)
    {
        sharedInput[localId] = d_input[threadId];
    }
    __syncthreads();

    if (threadId < d_numThreads)
    {
        // Perform a simple operation: sum the shared memory value 1000 times
        int sum = 0;
        for (int i = 0; i < 1000; i++)
        {
            sum += sharedInput[localId];
        }
        d_input[threadId] = sum; // Write back to global memory
    }
}

// Constant Memory Kernel: Uses constant memory for a comparison
__global__ void constantMemoryKernel(int* d_input)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < d_numThreads)
    {
        // Perform a simple operation: compare with constant_search_value
        int sum = 0;
        for (int i = 0; i < 1000; i++)
        {
            if (d_input[threadId] == constant_search_value)
            {
                sum += d_input[threadId];
            }
        }
        d_input[threadId] = sum; // Write back to global memory
    }
}

// Register Memory Kernel: Uses local variables (stored in registers)
__global__ void registerMemoryKernel(int* d_input, int d_numThreads)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < d_numThreads)
    {
        // Perform a simple operation: use a local variable (register)
        int localValue = d_input[threadId];
        int sum = 0;
        for (int i = 0; i < 1000; i++)
        {
            sum += localValue;
        }
        d_input[threadId] = sum; // Write back to global memory
    }
}

// Allocate pageable host memory (not pinned)
__host__ int* allocatePageableRandomHostMemory(int numElements)
{
    srand(time(0));
    size_t size = numElements * sizeof(int);

    // Allocate pageable memory using malloc
    int* data = (int*)malloc(size);
    if (data == nullptr)
    {
        fprintf(stderr, "Failed to allocate pageable host memory!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize with random values
    for (int i = 0; i < numElements; ++i)
    {
        data[i] = rand() % 255;
    }

    return data;
}

__host__ int* allocateDeviceMemory(int numElements)
{
    size_t size = numElements * sizeof(int);
    int* d_input = nullptr;
    cudaError_t err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return d_input;
}

__host__ void copyFromHostToDevice(std::string kernelType, int* input, int numElements, int numThreads, int* d_input)
{
    size_t size = numElements * sizeof(int);

    if (kernelType == "constant")
    {
        // Copy to constant memory
        int threadSpan = numElements / numThreads;
        cudaError_t err = cudaMemcpyToSymbol(d_numElements, &numElements, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy const numElements from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpyToSymbol(d_threadSpan, &threadSpan, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy const threadSpan from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaMemcpyToSymbol(d_numThreads, &numThreads, sizeof(int), 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy const numThreads from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        // Copy input to d_input
        cudaError_t err = cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy array input from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
}

__host__ void executeKernel(int* d_input, int numElements, int threadsPerBlock, std::string kernelType)
{
    int numThreads = numElements;
    int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock; // Ceiling division

    if (kernelType == "global")
    {
        globalMemoryKernel<<<numBlocks, threadsPerBlock>>>(d_input, numThreads);
    }
    else if (kernelType == "constant")
    {
        constantMemoryKernel<<<numBlocks, threadsPerBlock>>>(d_input);
    }
    else if (kernelType == "shared")
    {
        size_t sharedMemSize = threadsPerBlock * sizeof(int);
        sharedMemoryKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_input, numThreads);
    }
    else
    {
        registerMemoryKernel<<<numBlocks, threadsPerBlock>>>(d_input, numThreads);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch %s kernel (error code %s)!\n", kernelType.c_str(), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
}

__host__ void deallocateMemory(int* d_input)
{
    cudaError_t err = cudaFree(d_input);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ void cleanUpDevice()
{
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<int, std::string, int, std::string> parseCommandLineArguments(int argc, char* argv[])
{
    int elementsPerThread = 2;
    int threadsPerBlock = 256;
    std::string currentPartId = "test";
    std::string kernelType = "global";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option == "-t")
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option == "-m")
        {
            elementsPerThread = atoi(value.c_str());
        }
        else if (option == "-p")
        {
            currentPartId = value;
        }
        else if (option == "-k")
        {
            kernelType = value;
        }
    }

    return { elementsPerThread, currentPartId, threadsPerBlock, kernelType };
}

__host__ int* setUpInput(int numElements)
{
    srand(time(0));
    int* input;

    int searchValue = rand() % 255;
    cudaError_t err = cudaMemcpyToSymbol(constant_search_value, &searchValue, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy constant int d_v from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    input = allocatePageableRandomHostMemory(numElements);
    return input;
}

int main(int argc, char* argv[])
{
    auto [elementsPerThread, currentPartId, threadsPerBlock, kernelType] = parseCommandLineArguments(argc, argv);

    int numElements = elementsPerThread * threadsPerBlock;
    int* input = setUpInput(numElements);
    int* d_input = allocateDeviceMemory(numElements);

    copyFromHostToDevice(kernelType, input, numElements, threadsPerBlock, d_input);

    // Timing
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    executeKernel(d_input, numElements, threadsPerBlock, kernelType);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Clean up
    deallocateMemory(d_input);
    free(input); // Free pageable host memory

    // Write to output file
    std::ofstream outputfile("output.csv", std::ios_base::app);
    if (!outputfile.is_open())
    {
        fprintf(stderr, "Failed to open output.csv for writing!\n");
        exit(EXIT_FAILURE);
    }
    outputfile << currentPartId << "," << kernelType << "," << threadsPerBlock << "," << elementsPerThread << "," << elapsedTime << "\n";
    outputfile.close();

    cleanUpDevice();
    return 0;
}