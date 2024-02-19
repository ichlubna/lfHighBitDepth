#include <stdexcept>
#include <vector>
#include <glm/glm.hpp>
#include <string>
#include "methods.h"

class Interpolator
{
    public:
    class InterpolationParams
    {
        public:
        InterpolationParams* setOutputPath(std::string path)
        {
            outputPath = path; 
            return this;
        }

        InterpolationParams* setCoordinates(std::string inputCoordinates)
        {
            coordinates = parseCoordinates(inputCoordinates); 
            return this;
        }
        
        InterpolationParams* setScanRange(std::string range)
        {
            scanRange = parseCoordinates(range);
            //if(scanRange.x < 0 || scanRange.y < 0)
            //    throw std::runtime_error("Focusing scanning range cannot be negative!");
            if(scanRange.x > scanRange.y)
                throw std::runtime_error("The start of the scan range has to be lower than the end!");
            return this;
        }
       
        InterpolationParams* setParameter(float inputParameter)
        {
            parameter = inputParameter;
            return this;
        }
 
        InterpolationParams* setRuns(int runsCount)
        {
            if(runsCount < 0)
                throw std::runtime_error("Number of kernel runs cannot be negative!");
            else if(runsCount > 0)
                runs = runsCount;
            return this;
        }
 
        std::string outputPath;
        glm::vec2 coordinates;
        glm::vec2 scanRange{0,0.5};
        int runs{1};
        float parameter{0};
        
        private:
        glm::vec3 parseCoordinates(std::string coordinates) const;
        std::vector<std::string> split(std::string input, char delimiter='_') const;
    };

    Interpolator(std::string inputPath, float cameraSpacingAspect);
    ~Interpolator();
    void interpolate(InterpolationParams params);

    private:
    class TexturesInfo
    {
        public:
        glm::ivec2 colsRows;
        glm::ivec3 resolution;
    };
    enum KernelType{PROCESS};
    class KernelParams
    {
        public:
        void *data;
        int width;
        int height;
    };

    size_t DEVICE_CHANNELS{4};
    bool secondMapActive{false};
    bool useSecondaryFolder{false};
    bool useMips{false};
    bool useYUV{false};
    float inputCamerasSpacingAspect;
    std::vector<int*> surfaceInputArrays;
    std::vector<int*> surfaceOutputArrays;
    void *surfaceObjectsArr;
    void *textureObjectsArr;
    void *secondaryTextureObjectsArr;
    void *mipTextureObjectsArr;
    float *weightsGPU;
    size_t sharedSize{0};
    glm::ivec2 colsRows;
    glm::ivec2 resolution;
    std::string input;
    void init();
    void loadGPUOffsets(glm::vec2 viewCoordinates);
    void loadGPUData();
    TexturesInfo loadTextures(std::string input, void **textures);
    void loadGPUConstants(InterpolationParams params);
    void loadGPUWeights(glm::vec2 viewCoordinates);
    int* loadImageToArray(const float *data, glm::ivec3 size, bool copyFromDevice);
    void storeResults(std::string path);
    std::vector<float> generateWeights(glm::vec2 coords);
    std::pair<int, int*> createSurfaceObject(glm::ivec3 size, const float *data=nullptr, bool copyFromDevice=false);
    int createTextureObject(const float *data, glm::ivec3 size);
    void runKernel(KernelType, KernelParams={});
    void testKernel(KernelType kernel, std::string label, int runs);
};
