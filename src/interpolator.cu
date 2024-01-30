#include <stdexcept>
#define GLM_FORCE_SWIZZLE
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include "interpolator.h"
#include "kernels.cu"
#include "lfLoader.h"
#include "libs/loadingBar/loadingbar.hpp"

class Timer
{
    public:
    Timer()
    {    
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        cudaEventRecord(startEvent);
    }
    float stop()
    {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float time = 0;
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return time; 
    };
    private:
    cudaEvent_t startEvent, stopEvent;
};

Interpolator::Interpolator(std::string inputPath, float spacingAspect) : inputCamerasSpacingAspect{spacingAspect}, input{inputPath}
{
    init();
}

Interpolator::~Interpolator()
{
    cudaDeviceReset();
}

void Interpolator::init()
{
    loadGPUData();
    if(inputCamerasSpacingAspect == 0)
        inputCamerasSpacingAspect = 1;
    sharedSize = 0;//sizeof(half)*colsRows.x*colsRows.y;
}

int Interpolator::createTextureObject(const float *data, glm::ivec3 size)
{
    cudaTextureAddressMode cudaAddressMode = cudaAddressModeClamp;
    size_t pitch = size.x*size.z*sizeof(float);
    void *imageData;
    cudaMalloc(&imageData, size.x*size.y*size.z*sizeof(float));
    cudaMemcpy2D(imageData, pitch, data, pitch, pitch, size.y, cudaMemcpyHostToDevice);

    cudaChannelFormatDesc channels = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x, size.y);
    cudaMemcpy2DToArray(arr, 0, 0, imageData, pitch, pitch, size.y, cudaMemcpyDeviceToDevice);   
    cudaFree(imageData);
 
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = arr;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.filterMode = cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressMode;
    texDescr.addressMode[1] = cudaAddressMode;
    texDescr.readMode = cudaReadModeElementType;
    cudaTextureObject_t texObj{0};
    cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL);
    return texObj;
}

std::pair<int, int*> Interpolator::createSurfaceObject(glm::ivec3 size, const float *data, bool copyFromDevice)
{
    auto arr = loadImageToArray(data, size, copyFromDevice);
    cudaResourceDesc surfRes;
    memset(&surfRes, 0, sizeof(cudaResourceDesc));
    surfRes.resType = cudaResourceTypeArray;
    surfRes.res.array.array = reinterpret_cast<cudaArray*>(arr);
    cudaSurfaceObject_t surfObj = 0;
    cudaCreateSurfaceObject(&surfObj, &surfRes);
    return {surfObj, arr};
}

int* Interpolator::loadImageToArray(const float *data, glm::ivec3 size, bool copyFromDevice)
{
    cudaChannelFormatDesc channels = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray *arr;
    cudaMallocArray(&arr, &channels, size.x*sizeof(float), size.y, cudaArraySurfaceLoadStore);
    if(data != nullptr)
    {
        if(copyFromDevice)
            cudaMemcpy2DArrayToArray(arr, 0, 0, reinterpret_cast<cudaArray_const_t>(data), 0, 0, size.x*sizeof(float), size.y, cudaMemcpyDeviceToDevice); 
        else
            cudaMemcpy2DToArray(arr, 0, 0, data, size.x*size.z, size.x*size.z, size.y, cudaMemcpyHostToDevice);
    }
    return reinterpret_cast<int*>(arr);
}

Interpolator::TexturesInfo Interpolator::loadTextures(std::string input, void **textures)
{
    LfLoader lfLoader;
    lfLoader.loadData(input);
    auto textColsRows = lfLoader.getColsRows();
    auto textResolution = lfLoader.imageResolution();

    std::cout << "Uploading content of "+input+" to GPU..." << std::endl;
    LoadingBar bar(lfLoader.imageCount());
   
    std::vector<cudaTextureObject_t> textureObjects;
    for(int col=0; col<textColsRows.x; col++)
        for(int row=0; row<textColsRows.y; row++)
        { 
            textureObjects.push_back(createTextureObject(lfLoader.image({col, row}).data(), textResolution)); 
            bar.add();
        }

    cudaMalloc(textures, textureObjects.size()*sizeof(cudaTextureObject_t));
    cudaMemcpy(*textures, textureObjects.data(), textureObjects.size()*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice); 
    return {textColsRows, textResolution};
}

void Interpolator::loadGPUData()
{
    std::string path = input;
    if(path.back() == '/')
        path.pop_back();

    auto info = loadTextures(input, &textureObjectsArr);
    colsRows = info.colsRows;
    resolution = info.resolution;
 
    std::vector<cudaSurfaceObject_t> surfaces;
    for(int i=0; i<FileNames::OUTPUT_COUNT; i++)
    {
        auto surfaceRes = resolution;
        auto surface = createSurfaceObject({surfaceRes, DEVICE_CHANNELS});
        surfaces.push_back(surface.first);  
        surfaceOutputArrays.push_back(surface.second);
    }
    cudaMalloc(&surfaceObjectsArr, surfaces.size()*sizeof(cudaTextureObject_t));
    cudaMemcpy(surfaceObjectsArr, surfaces.data(), surfaces.size()*sizeof(cudaSurfaceObject_t), cudaMemcpyHostToDevice);
}

void Interpolator::loadGPUConstants(InterpolationParams params)
{
    std::vector<int> intValues(IntConstantIDs::INT_CONSTANTS_COUNT);
    intValues[IntConstantIDs::IMG_RES_X] = resolution.x;
    intValues[IntConstantIDs::IMG_RES_Y] = resolution.y;
    intValues[IntConstantIDs::GRID_SIZE] = colsRows.x*colsRows.y;
    intValues[IntConstantIDs::COLS] = colsRows.x;
    intValues[IntConstantIDs::ROWS] = colsRows.y;
    cudaMemcpyToSymbol(Kernels::Constants::intConstants, intValues.data(), intValues.size() * sizeof(int));
    
    std::vector<float> floatValues(FloatConstantIDs::FLOAT_CONSTANTS_COUNT);
    float rangeSize = params.scanRange.y - params.scanRange.x; 
    floatValues[FloatConstantIDs::SCAN_RANGE_SIZE] = rangeSize;
    floatValues[FloatConstantIDs::SCAN_RANGE_START] = params.scanRange.x;
    floatValues[FloatConstantIDs::SCAN_RANGE_END] = params.scanRange.y;
    float2 pixelSize{1.0f/resolution.x, 1.0f/resolution.y}; 
    floatValues[FloatConstantIDs::PX_SIZE_X] = pixelSize.x;
    floatValues[FloatConstantIDs::PX_SIZE_Y] = pixelSize.y;
    floatValues[FloatConstantIDs::PX_SIZE_X_HALF] = pixelSize.x/2.0f;
    floatValues[FloatConstantIDs::PX_SIZE_Y_HALF] = pixelSize.y/2.0f;
    cudaMemcpyToSymbol(Kernels::Constants::floatConstants, floatValues.data(), floatValues.size() * sizeof(float));

    std::vector<void*> dataPointers(DataPointersIDs::POINTERS_COUNT);
    dataPointers[DataPointersIDs::SURFACES] = reinterpret_cast<void*>(surfaceObjectsArr);
    dataPointers[DataPointersIDs::TEXTURES] = reinterpret_cast<void*>(textureObjectsArr);
    cudaMemcpyToSymbol(Kernels::Constants::dataPointers, dataPointers.data(), dataPointers.size() * sizeof(void*));
            
    int blockSampling = 1; 
    float2 pixelSizeBlock{blockSampling*pixelSize.x, blockSampling*pixelSize.y}; 
    std::vector<float2> blockOffsets{ {0.0f, 0.0f}, {-1.0f*pixelSizeBlock.x, 0.5f*pixelSizeBlock.y}, {0.5f*pixelSizeBlock.x, 1.0f*pixelSizeBlock.y}, {1.0f*pixelSizeBlock.x, -0.5f*pixelSizeBlock.y}, {-0.5f*pixelSizeBlock.x, -1.0f*pixelSizeBlock.y} };
    cudaMemcpyToSymbol(Kernels::Constants::blockOffsets, blockOffsets.data(), BLOCK_OFFSET_COUNT * sizeof(float2));
}

void Interpolator::loadGPUOffsets(glm::vec2 viewCoordinates)
{
    float aspect = (static_cast<float>(resolution.x)/resolution.y) / inputCamerasSpacingAspect;
    std::vector<float2> offsets(colsRows.x*colsRows.y);
    int gridID = 0; 
    for(int row=0; row<colsRows.y; row++) 
    {     
        gridID = row*colsRows.x;
        for(int col=0; col<colsRows.x; col++) 
        {
            float2 offset{(viewCoordinates.x-col)/colsRows.x, (viewCoordinates.y-row)/colsRows.y};
            offset.y *= aspect;
            offsets[gridID] = offset;
            gridID++;
        }
    }
    cudaMemcpyToSymbol(Kernels::Constants::offsets, offsets.data(), offsets.size() * sizeof(float2));
}

std::vector<float> Interpolator::generateWeights(glm::vec2 coords)
{
    auto maxDistance = glm::distance(glm::vec2(0,0), glm::vec2(colsRows));
    float weightSum{0};
    std::vector<float> weightVals;
    for(int row=0; row<colsRows.y; row++) 
        for(int col=0; col<colsRows.x; col++) 
        {
            float weight = maxDistance - glm::distance(coords, glm::vec2(col, row));
            weightSum += weight;
            weightVals.push_back(weight);
        }
    for(auto &weight : weightVals)
        weight /= weightSum; 
    return weightVals;
}


void Interpolator::loadGPUWeights(glm::vec2 viewCoordinates)
{
    auto weights = generateWeights(viewCoordinates);
    cudaMemcpyToSymbol(Kernels::Constants::weights, weights.data(), weights.size() * sizeof(float));
}

std::vector<std::string> Interpolator::InterpolationParams::split(std::string input, char delimiter) const
{
    std::vector<std::string> values;
    std::stringstream ss(input); 
    std::string value; 
    while(getline(ss, value, delimiter))
        values.push_back(value);
    return values;
}

glm::vec3 Interpolator::InterpolationParams::parseCoordinates(std::string coordinates) const
{
    auto values = split(coordinates);
    glm::vec3 numbers{0};
    int i{0};
    for(auto& value : values)
    {
        if(i>3)
            throw std::runtime_error("Coordinates too long!");
        if(value[0] == '#')
        {
            value.erase(0,1);
            auto newValue = "0x"+value;
            numbers[i] = std::stoul(newValue, nullptr, 16); 
        }
        else
            numbers[i] = std::stof(value);
        i++;
    }

    return numbers;
}

void Interpolator::runKernel(KernelType type, KernelParams params)
{
    dim3 dimBlock(16, 16, 1);
    constexpr size_t SUBSAMPLING{2};
    switch(type)
    {
        case PROCESS:
        {
            dim3 dimGrid(glm::ceil(static_cast<float>(resolution.x/SUBSAMPLING)/dimBlock.x), glm::ceil(static_cast<float>(resolution.y/SUBSAMPLING)/dimBlock.y), 1);
            Kernels::process<<<dimGrid, dimBlock, sharedSize>>>();
        }
        break;
    }
    cudaDeviceSynchronize();
    //std::cerr << "Error: " << cudaPeekAtLastError() << std::endl;
}

void Interpolator::testKernel(KernelType kernel, std::string label, int runs)
{
    std::cout << "Elapsed time of "+label+": "<<std::endl;
    float avgTime{0};
    for(int i=0; i<runs; i++)
    {
        Timer timer;
        runKernel(kernel);
        auto time = timer.stop();
        avgTime += time;
        std::cout << "Run #" << i<< ": " << time << " ms" << std::endl;
    }
    std::cout << "Average of " << runs << " runs of "+label+": " << avgTime/runs << " ms" << std::endl;
}

void Interpolator::interpolate(InterpolationParams params)
{
    glm::vec2 coords = glm::vec2(colsRows-1)*params.coordinates;
    loadGPUWeights(coords);
    loadGPUOffsets(coords);   
    loadGPUConstants(params);
    
    testKernel(PROCESS, "interpolation", params.runs);
    storeResults(params.outputPath);
}


void Interpolator::storeResults(std::string path)
{
    std::cout << "Storing results..." << std::endl;
    LoadingBar bar(FileNames::OUTPUT_COUNT-1);
    std::vector<float> data(resolution.x*resolution.y*DEVICE_CHANNELS, 255);

    size_t pitch = resolution.x*DEVICE_CHANNELS*sizeof(float);
    void *imageData;
    cudaMalloc(&imageData, resolution.x*resolution.y*DEVICE_CHANNELS*sizeof(float));

    for(int i=0; i<FileNames::OUTPUT_COUNT; i++) 
    {
        cudaMemcpy2DFromArray(imageData, pitch, reinterpret_cast<cudaArray*>(surfaceOutputArrays[i]), 0, 0, pitch, resolution.y, cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(data.data(), pitch, imageData, pitch, pitch, resolution.y, cudaMemcpyDeviceToHost);
        LfLoader::saveOutput(data.data(), resolution.x, resolution.y, std::filesystem::path(path)/"render.exr");
        bar.add();
    }
    cudaFree(imageData);
}
