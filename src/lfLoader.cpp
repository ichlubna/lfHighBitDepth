#include <stdexcept>
#define TINYEXR_USE_MINIZ 0
#include <zlib.h>
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
#include <cstring>
#include "libs/loadingBar/loadingbar.hpp"
#include "lfLoader.h"

const std::set<std::filesystem::path> LfLoader::listPath(std::string path) const
{
    if(!std::filesystem::exists(path))
        throw std::runtime_error("The path "+path+" does not exist!");
    if(!std::filesystem::is_directory(path))
        throw std::runtime_error("The path "+path+" does not lead to a directory!");

    auto dir = std::filesystem::directory_iterator(path);
    std::set<std::filesystem::path> sorted;
    for(const auto &file : dir)
        sorted.insert(file.path().filename());
    return sorted;
}

glm::ivec2 LfLoader::parseFilename(std::string name) const
{
    auto delimiterPos = name.find('_');
    if(delimiterPos == std::string::npos)
        throw std::runtime_error("File "+name+" is not named properly as column_row.extension!");
    int extensionPos = name.find('.');
    auto row = name.substr(0, delimiterPos);
    auto col = name.substr(delimiterPos + 1, extensionPos - delimiterPos - 1);
    return {stoi(row), stoi(col)};
}

void LfLoader::loadImage(std::string path, glm::uvec2 coords)
{
    float *pixels;
    const char *error{nullptr};
    int ret = LoadEXR(&pixels, &resolution.x, &resolution.y, path.c_str(), &error);
    resolution.z = 3;

    if (ret != TINYEXR_SUCCESS)
    {
       throw std::runtime_error("Cannot open image " + path + " - " + error);
       FreeEXRErrorMessage(error);
    }
    size_t size = resolution.x*resolution.y*resolution.z;
    grid[coords.x][coords.y] = std::vector<float>(pixels, pixels+size);
}

void LfLoader::initGrid(glm::uvec2 inColsRows)
{
    colsRows = inColsRows+glm::uvec2(1);
    grid.resize(colsRows.x);
    for(auto &row : grid)
        row.resize(colsRows.y);
}

void LfLoader::loadData(std::string path)
{  
    auto files = listPath(path);
    if(files.empty())
        throw std::runtime_error("The input directory is empty!");
    initGrid(glm::uvec2(2,2));
    if(files.size() != 4)
        throw std::runtime_error("The number of input files is wrong. Expecting 4 input views (2x2 grid).");

    std::cout << "Loading images..." << std::endl;
    LoadingBar bar(files.size());
    int i{0};
    for(auto const &file : files)
    {
        i++;
        int row{i/2};
        int col{i%2};        
        loadImage(path/file, {row, col}); 
        bar.add(); 
    }
}

void LfLoader::saveOutput(const float* rgb, int width, int height, std::string filename) {

    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);

    constexpr size_t INPUT_CHANNELS{4};
    // Split RGBRGBRGB... into R, G and B layer
    for (int i = 0; i < width * height; i++) {
      images[0][i] = rgb[INPUT_CHANNELS*i+0];
      images[1][i] = rgb[INPUT_CHANNELS*i+1];
      images[2][i] = rgb[INPUT_CHANNELS*i+2];
    }

    float* image_ptr[3];
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
    strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
    strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
      header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
      header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }

    const char* err = NULL; // or nullptr in C++11 or later.
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
      fprintf(stderr, "Save EXR err: %s\n", err);
      FreeEXRErrorMessage(err); // free's buffer for an error message
    }

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

  }
