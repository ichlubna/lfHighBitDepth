#include <glm/glm.hpp>
#include <curand_kernel.h>
#include "methods.h"

namespace Kernels
{
    constexpr int BLOCK_SAMPLE_COUNT{5};
    constexpr int PIXEL_SAMPLE_COUNT{1};
    constexpr int CLOSEST_COUNT{4};

    namespace Constants
    {
        __constant__ int intConstants[IntConstantIDs::INT_CONSTANTS_COUNT];
        __device__ int2 imgRes(){return {intConstants[IntConstantIDs::IMG_RES_X], intConstants[IntConstantIDs::IMG_RES_Y]};}
        __device__ int2 colsRows(){return{intConstants[IntConstantIDs::COLS], intConstants[IntConstantIDs::ROWS]};}
        __device__ int gridSize(){return intConstants[IntConstantIDs::GRID_SIZE];} 
        __constant__ void* dataPointers[DataPointersIDs::POINTERS_COUNT];
        __device__ cudaSurfaceObject_t* surfaces(){return reinterpret_cast<cudaSurfaceObject_t*>(dataPointers[DataPointersIDs::SURFACES]);}
        __device__ cudaTextureObject_t* textures(){return reinterpret_cast<cudaTextureObject_t*>(dataPointers[DataPointersIDs::TEXTURES]);} 
        
        __constant__ float floatConstants[FloatConstantIDs::FLOAT_CONSTANTS_COUNT];
        __device__ float scanStart(int i){return floatConstants[FloatConstantIDs::SCAN_RANGE_START_1+i];}
        __device__ float scanStep(int i){return floatConstants[FloatConstantIDs::SCAN_RANGE_STEP_1+i];}
        __device__ float2 halfPixelSize(){return {floatConstants[FloatConstantIDs::PX_SIZE_X_HALF], floatConstants[FloatConstantIDs::PX_SIZE_Y_HALF]};}
        __device__ float parameter(){return floatConstants[FloatConstantIDs::PARAMETER];}

        __device__ constexpr int MAX_IMAGES{4};
        __constant__ float weights[MAX_IMAGES];
        __constant__ float2 offsets[MAX_IMAGES];
        __constant__ int2 blockOffsets[BLOCK_OFFSET_COUNT];
    }

    //extern __shared__ half localMemory[];

    __device__ float4 operator+ (const float4  &a, const float4 &b)
    {
        float4 result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        result.z = a.z + b.z;
        return result;
    }

    __device__ int2 operator+ (const int2  &a, const int2 &b)
    {
        int2 result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        return result;
    }
    
    __device__ float4 operator/ (const float4  &a, const float &b)
    {
        float4 result;
        result.x = a.x/b;
        result.y = a.y/b;
        result.z = a.z/b;
        return result;
    }
    
    __device__ float4 operator* (const float  &a, const float4 &b)
    {
        float4 result;
        result.x = a*b.x;
        result.y = a*b.y;
        result.z = a*b.z;
        return result;
    }

    __device__ bool coordsOutside(int2 coords, int2 resolution)
    {
        return (coords.x >= resolution.x || coords.y >= resolution.y);
    }

    __device__ int2 getImgCoords()
    {
        int2 coords;
        constexpr int SUBSAMPLING{2};
        coords.x = (threadIdx.x + blockIdx.x * blockDim.x)*SUBSAMPLING;
        coords.y = (threadIdx.y + blockIdx.y * blockDim.y)*SUBSAMPLING;
        return coords;
    }

    __device__ float2 normalizeCoords(int2 coords)
    {
        auto res = Constants::imgRes();
        return {static_cast<float>(coords.x)/res.x,
                static_cast<float>(coords.y)/res.y};
    }
   
    namespace Pixel
    {
        __device__ float distance(float3 &a, float3 &b)
        {
            float dist = fmaxf(fmaxf(fabsf(a.x-b.x), fabsf(a.y-b.y)), fabsf(a.z-b.z));
            return dist;
        }

        __device__ void store(float4 px, int imageID, int2 coords)
        {
            surf2Dwrite<float4>(px, Constants::surfaces()[imageID], coords.x*sizeof(float4), coords.y);
        } 
        
        __device__ float4 load(int imageID, int2 coords)
        {
            int id = Constants::textures()[imageID];
            float2 halfPx = Constants::halfPixelSize(); 
            //return tex2D<float4>(id, coords.x+halfPx.x, coords.y+halfPx.y);
            return tex2D<float4>(id, coords.x, coords.y);
        } 

        __device__ float4 gammaCorrect(float4 color)
        {
            constexpr float GAMMA{2.2};
            float4 outColor;
            float *outColorArr = reinterpret_cast<float*>(&outColor);
            float *colorArr = reinterpret_cast<float*>(&color);
            for(int channel=0; channel<CHANNEL_COUNT; channel++)
            {
                float gammaCorrected = powf(colorArr[channel], 1.0f / GAMMA);
                outColorArr[channel] = min(1.0f, max(0.0f, gammaCorrected));
            }
            return outColor;
        }
       
        __device__ float4 quantize(float4 color, float maxValue)
        {
            float4 outColor;
            float *outColorArr = reinterpret_cast<float*>(&outColor);
            float *colorArr = reinterpret_cast<float*>(&color);
            for(int channel=0; channel<CHANNEL_COUNT; channel++)
            {
                outColorArr[channel] = round(colorArr[channel] * maxValue);
            }
            return outColor;
        }

        // shift from 0.5 to 0.9
        __device__ float logFormula(float value, float shift)
        {
            return logf(value) + shift;
        }
       
        //  order from 0.1 to 0.9
        __device__ float exponentialFormula(float value, float order)
        {
            return powf(value, order);
        }
        
        //  order from 2 to 8 with step 0.4 
        __device__ float exponentialOddFormula(float value, float order)
        {
            float o = static_cast<int>(round(order))*0.4;
            return powf(value-1, o) + 1;
        }

        __device__ float4 colorProfile(float4 color)
        {
            float4 outColor;
            float *outColorArr = reinterpret_cast<float*>(&outColor);
            float *colorArr = reinterpret_cast<float*>(&color);
            for(int channel=0; channel<CHANNEL_COUNT; channel++)
            {
                outColorArr[channel] = colorArr[channel];
            }
            return outColor;
        }

        __device__ float4 transform(float4 color, int pixelID)
        {
            Constants::parameter();
            return color;
        }
    }
 
        class ElementRange
        {
            private:
            float3 minCol{FLT_MAX, FLT_MAX, FLT_MAX};
            float3 maxCol{FLT_MIN, FLT_MIN, FLT_MIN};
            
            public:
            __device__ void add(float4 val)
            {
                minCol.x = fminf(minCol.x, val.x);
                minCol.y = fminf(minCol.y, val.y);
                minCol.z = fminf(minCol.z, val.z);
                maxCol.x = fmaxf(maxCol.x, val.x);
                maxCol.y = fmaxf(maxCol.y, val.y);
                maxCol.z = fmaxf(maxCol.z, val.z);
            }
            __device__ float dispersionAmount()
            {
                return Pixel::distance(minCol, maxCol); 
            }      
            __device__ ElementRange& operator+=(const float4& rhs){

              add(rhs);
              return *this;
            }
        };

    __device__ int2 focusCoords(int gridID, int2 pxCoords, float focus)
    {
        float2 offset = Constants::offsets[gridID];
        //return {static_cast<int>(round(offset.x*focus+pxCoords.x)), static_cast<int>(round(offset.y*focus+pxCoords.y))};
        return {__float2int_rn(__fmaf_rn(offset.x, focus, pxCoords.x)),__float2int_rn(__fmaf_rn(offset.y, focus, pxCoords.y))};
    }

    namespace FocusLevel
    {      
        __device__ void evaluateBlock(int gridID, float focus, int2 coords, ElementRange *dispersions, int pixelID)
        {
            for(int blockPx=0; blockPx<BLOCK_OFFSET_COUNT; blockPx++)
            {
                int2 offset = Constants::blockOffsets[blockPx]; 
                int2 inBlockCoords{coords.x+offset.x, coords.y+offset.y};
                auto px{Pixel::load(gridID, focusCoords(gridID, inBlockCoords, focus))};
                px = Pixel::transform(px, pixelID);
                dispersions[blockPx] += px;
            }
        }

        __device__ float evaluateDispersion(int2 coords, float focus, int pixelID)
        {
            auto cr = Constants::colsRows();
            ElementRange dispersionCalc[BLOCK_OFFSET_COUNT];
                
            int gridID = 0;
            for(int row=0; row<cr.y; row++) 
            {     
                gridID = row*cr.x;
                for(int col=0; col<cr.x; col++) 
                {
                    evaluateBlock(gridID, focus, coords, dispersionCalc, pixelID);
                    gridID++;
                }
            } 
            float finalDispersion{0};
            for(int blockPx=0; blockPx<BLOCK_OFFSET_COUNT; blockPx++)
                finalDispersion += dispersionCalc[blockPx].dispersionAmount();
            return finalDispersion;
        }
 
        __device__ float4 render(int2 coords, float focus)
        {
            auto cr = Constants::colsRows();
            float4 sum{0,0,0,0};
            int gridID = 0;  
            auto weights = Constants::weights;
            for(int row=0; row<cr.y; row++) 
            {     
                gridID = row*cr.x;
                for(int col=0; col<cr.x; col++) 
                {
                    auto px{Pixel::load(gridID, focusCoords(gridID, coords, focus))};
                    sum = sum + weights[gridID] * px;
                    gridID++;
                }
            }
            sum.w = 1.0f;
            return sum;
        }      
    }
    
    namespace Focusing
    {    
        class Optimum
        {
            public:
            float optimalFocus{0};
            float minDispersion{FLT_MAX};
            __device__ bool add(float focus, float dispersion)
            {
                if(dispersion < minDispersion)
                {
                   minDispersion = dispersion;
                   optimalFocus = focus; 
                   return true;
                }
                return false;
            }
            __device__ void addForce(float focus, float dispersion)
            {
                   minDispersion = dispersion;
                   optimalFocus = focus; 
            }
        }; 

        __device__ Optimum& minOpt(Optimum &a, Optimum &b)
        {
            if(a.minDispersion < b.minDispersion)
                return a;
            else
                return b;
        }

        __device__ float bruteForce(int2 coords, int pixelID)
        {
            float focus{Constants::scanStart(pixelID)};
            float stepSize{Constants::scanStep(pixelID)};
            Optimum optimum;
            
            for(int step=0; step<FOCUS_STEPS_COUNT; step++)
            {
                float dispersion = FocusLevel::evaluateDispersion(coords, focus, pixelID);
                optimum.add(focus, dispersion);
                focus += stepSize;  
            }
            return optimum.optimalFocus;
        }
    }

    __device__ void focusAndRender(int2 cornerCoords)
    {
        int2 coords = cornerCoords;
        int pixelID{0};
        float focus{0};
        for(int x=0; x<2; x++)
            for(int y=0; y<2; y++)
            {
                coords = cornerCoords + int2{x, y};
                focus += Focusing::bruteForce(coords, pixelID);
                pixelID++;
            }
        focus /= 4;
        for(int x=0; x<2; x++)
            for(int y=0; y<2; y++)
            {
                coords = cornerCoords + int2{x, y};
                auto color = FocusLevel::render(coords, focus);
                Pixel::store(color, FileNames::RENDER_IMAGE, coords);
            }
    }

    __global__ void process()
    {
        int2 coords = getImgCoords();
        if(coordsOutside(coords, Constants::imgRes()))
            return;
        focusAndRender(coords);   
    }
}
