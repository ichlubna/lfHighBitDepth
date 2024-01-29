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
        __device__ float2 scanRange(){return {floatConstants[FloatConstantIDs::SCAN_RANGE_START], floatConstants[FloatConstantIDs::SCAN_RANGE_END]};}
        __device__ float scanRangeSize(){return floatConstants[FloatConstantIDs::SCAN_RANGE_SIZE];}
        __device__ float2 pixelSize(){return {floatConstants[FloatConstantIDs::PX_SIZE_X], floatConstants[FloatConstantIDs::PX_SIZE_Y]};}
        __device__ float2 halfPixelSize(){return {floatConstants[FloatConstantIDs::PX_SIZE_X_HALF], floatConstants[FloatConstantIDs::PX_SIZE_Y_HALF]};}

        __device__ constexpr int MAX_IMAGES{4};
        __constant__ float weights[MAX_IMAGES];
        __constant__ float2 offsets[MAX_IMAGES];
        __constant__ float2 blockOffsets[BLOCK_OFFSET_COUNT];
    }

    //extern __shared__ half localMemory[];

    __device__ bool coordsOutside(int2 coords, int2 resolution)
    {
        return (coords.x >= resolution.x || coords.y >= resolution.y);
    }

    __device__ int2 getImgCoords()
    {
        int2 coords;
        coords.x = (threadIdx.x + blockIdx.x * blockDim.x);
        coords.y = (threadIdx.y + blockIdx.y * blockDim.y);
        return coords;
    }

    __device__ float2 normalizeCoords(int2 coords)
    {
        auto res = Constants::imgRes();
        return {static_cast<float>(coords.x)/res.x,
                static_cast<float>(coords.y)/res.y};
    }
   
    namespace Pixel
    {/*
        __device__ float distance(PixelArray &a, PixelArray &b)
        {
            float dist = fmaxf(fmaxf(fabsf(a[0]-b[0]), fabsf(a[1]-b[1])), fabsf(a[2]-b[2]));
            return dist;
        }
*/
        __device__ void store(float4 px, int imageID, int2 coords)
        {
            surf2Dwrite<float4>(px, Constants::surfaces()[imageID], coords.x*sizeof(float4), coords.y);
        }
        
        
        __device__ float4 load(int imageID, float2 coords)
        {
            int id = Constants::textures()[imageID];
            float2 halfPx = Constants::halfPixelSize(); 
            return tex2D<float4>(id, coords.x+halfPx.x, coords.y+halfPx.y);
        }
    }
/* 
        class ElementRange
        {
            private:
            PixelArray minCol{float4{FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX}};
            PixelArray maxCol{float4{FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN}};
            
            public:
            __device__ void add(PixelArray val)
            {
                minCol[0] = fminf(minCol[0],val[0]);
                minCol[1] = fminf(minCol[1],val[1]);
                minCol[2] = fminf(minCol[2],val[2]);
                maxCol[0] = fmaxf(maxCol[0],val[0]);
                maxCol[1] = fmaxf(maxCol[1],val[1]);
                maxCol[2] = fmaxf(maxCol[2],val[2]);
            }
            __device__ float dispersionAmount()
            {
                return Pixel::distance(minCol, maxCol); 
            }      
            __device__ ElementRange& operator+=(const PixelArray& rhs){

              add(rhs);
              return *this;
            }
        };


    __device__ float2 focusCoords(int gridID, float2 pxCoords, float focus)
    {
        float2 offset = Constants::offsets[gridID];
        //float2 coords{offset.x*focus+pxCoords.x, offset.y*focus+pxCoords.y};
        float2 coords{__fmaf_rn(offset.x, focus, pxCoords.x), __fmaf_rn(offset.y, focus, pxCoords.y)};
        return coords;
    }

    namespace FocusLevel
    {      
        template<int blockSize, typename T> 
        __device__ void evaluateBlock(int gridID, float focus, float2 coords, T *dispersions)
        {
            for(int blockPx=0; blockPx<blockSize; blockPx++)
            {
                float2 offset = Constants::blockOffsets[blockPx]; 
                float2 inBlockCoords{coords.x+offset.x, coords.y+offset.y};
                auto px{Pixel::load(gridID, focusCoords(gridID, inBlockCoords, focus))};
                dispersions[blockPx] += px;
            }
        }

        template<typename T, int blockSize>
        __device__ float evaluateDispersion(float2 coords, float focus)
        {
            auto cr = Constants::colsRows();
            T dispersionCalc[blockSize];
                
            int gridID = 0;
            for(int row=0; row<cr.y; row++) 
            {     
                gridID = row*cr.x;
                for(int col=0; col<cr.x; col++) 
                {
                    evaluateBlock<blockSize>(gridID, focus, coords, dispersionCalc);
                    gridID++;
                }
            } 
            float finalDispersion{0};
            for(int blockPx=0; blockPx<blockSize; blockPx++)
                finalDispersion += dispersionCalc[blockPx].dispersionAmount();
            return finalDispersion;
        }

       
        template<DispersionMode mode>
        __device__ uchar4 render(float2 coords, float focus)
        {
            auto cr = Constants::colsRows();
            PixelArray sum;
            int gridID = 0; 
          
                auto weights = Constants::weights;
                for(int row=0; row<cr.y; row++) 
                {     
                    gridID = row*cr.x;
                    for(int col=0; col<cr.x; col++) 
                    {
                        auto px{Pixel::load(gridID, focusCoords(gridID, coords, focus))};
                        sum.addWeighted(weights[gridID], px);
                        gridID++;
                    }
                }

            return sum.uch4();
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

        __device__ float bruteForce(float2 coords)
        {
            int steps = 32;
            float stepSize{static_cast<float>(Constants::scanRangeSize())/steps};
            float focus{Constants::scanRange().x};
            Optimum optimum;
            
            int wasMin{0};
            int terminateIn{steps>>2}; 
            for(int step=0; step<steps; step++)
            {
                focus += stepSize;  
            }
            return optimum.optimalFocus;
        }

    }
*/
    __global__ void process()
    {
        int2 threadCoords = getImgCoords();
        if(coordsOutside(threadCoords, Constants::imgRes()))
            return;
        auto coords = normalizeCoords(threadCoords); 

        float focus{0};
        
        //auto color = Pixel::load(0, threadCoords);
        float4 color{1.0, 0.5, 0.7, 0.3};
        Pixel::store(color, FileNames::RENDER_IMAGE, threadCoords);
    }

}
