#include <metal_stdlib>

using namespace metal;

typedef struct {
    unsigned long width, height, depth;
} MTLSize;

kernel void conv(
    device float *inputData [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *biasesData [[ buffer(2) ]],
    device float *outputData [[ buffer(3) ]],

    constant MTLSize* iDims [[ buffer(4) ]],
    constant MTLSize* wDims [[ buffer(5) ]],
    constant MTLSize* oDims [[ buffer(6) ]],

    constant uint& filtersCount [[ buffer(7) ]],
    constant uint& batchSize [[ buffer(8) ]],
    constant uint& padding [[ buffer(9) ]],
    constant uint& stride [[ buffer(10) ]],

    const uint3 outputPos [[ thread_position_in_grid ]] )
{
    uint batchIndex = outputPos.z / filtersCount;

    uint imageDepth = iDims->depth / batchSize;
    uint imageSize = iDims->width * iDims->height * imageDepth;
    uint imageSquare = iDims->height * iDims->width;

    uint filterIndex = outputPos.z - (batchIndex * filtersCount);
    uint filterSize = wDims->width * wDims->height * imageDepth;
    uint filterSquare = wDims->height * wDims->width;

    // For each filter we use it's own bias from biasesData.
    float s = biasesData[filterIndex];

    for (uint z = 0; z < imageDepth; ++z) {
        // z = filterChannel or imageChannel
        uint filterOffset = filterIndex * filterSize + z * filterSquare;
        uint imageOffset = batchIndex * imageSize + z * imageSquare;

        for (uint y = 0; y < wDims->height; ++y) {
            int iy = int(outputPos.y) * int(stride) - int(padding) + int(y);
            if (iy < 0 || iy >= iDims->height) {
                continue;
            }
            for (uint x = 0; x < wDims->width; ++x) {
                int ix = int(outputPos.x) * int(stride) - int(padding) + int(x);
                if (ix > -1 && ix < iDims->width) {
                    float weight = weightsData[filterOffset + y * wDims->width + x];
                    float input = inputData[imageOffset + iy * iDims->width + ix];

                    s += weight * input;
                }
            }
        }
    }

    outputData[outputPos.z * oDims->width * oDims->height + outputPos.y * oDims->width + outputPos.x] = s;
}

kernel void calcInputGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],

    constant MTLSize* iDims [[ buffer(3) ]],
    constant MTLSize* wDims [[ buffer(4) ]],
    constant MTLSize* oDims [[ buffer(5) ]],

    constant uint& filtersCount [[ buffer(6) ]],
    constant uint& batchSize [[ buffer(7) ]],
    constant uint& padding [[ buffer(8) ]],
    constant uint& stride [[ buffer(9) ]],

    const uint3 inputPos [[ thread_position_in_grid ]] )
{
    uint imageDepth = iDims->depth / batchSize;
    uint batchIndex = inputPos.z / imageDepth;
    uint imageChannel = inputPos.z - (batchIndex * imageDepth);

    float s = 0;

    uint filterSquare = wDims->width * wDims->height;
    uint filterSize = filterSquare * imageDepth;
    uint outputSquare = oDims->width * oDims->height;

    for (int filterIndex = 0; filterIndex < filtersCount; ++filterIndex) {
        int filterOffset = filterIndex * filterSize + imageChannel * filterSquare;
        int outputOffset = batchIndex * filtersCount * outputSquare + filterIndex * outputSquare;

        for (int y = 0; y < wDims->height; ++y) {
            int wy = wDims->height - 1 - y; // weight rotation
            int oyNumer = int(inputPos.y) + int(padding) - y;
            if (oyNumer % int(stride) != 0) {
                continue;
            }
            int oy = oyNumer / int(stride);
            if (oy < 0 || oy >= oDims->height) {
                continue;
            }
            for (int x = 0; x < wDims->width; ++x) {
                int wx = wDims->width - 1 - x; // weight rotation
                int oxNumer = int(inputPos.x) + int(padding) - x;
                if (oxNumer % int(stride) != 0) {
                    continue;
                }
                int ox = oxNumer / int(stride);
                if (ox > -1 && ox < oDims->width) {
                    float weight = weightsData[filterOffset + wy * wDims->width + wx];
                    float gradient = outputGrad[outputOffset + oy * oDims->width + ox];
                    s += weight * gradient;
                }
            }
        }
    }

    inputGrad[inputPos.z * iDims->width * iDims->height + inputPos.y * iDims->width + inputPos.x] += s;
}

kernel void calcWeightGrads(
    device float *inputData [[ buffer(0) ]],
    device float *weightsGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],

    constant MTLSize* iDims [[ buffer(3) ]],
    constant MTLSize* wDims [[ buffer(4) ]],
    constant MTLSize* oDims [[ buffer(5) ]],

    constant uint& filtersCount [[ buffer(6) ]],
    constant uint& batchSize [[ buffer(7) ]],
    constant uint& padding [[ buffer(8) ]],
    constant uint& stride [[ buffer(9) ]],

    const uint3 filterPos [[ thread_position_in_grid ]] )
{
    uint imageDepth = iDims->depth / batchSize;
    uint imageSquare = iDims->width * iDims->height;
    uint filterIndex = filterPos.z / imageDepth;
    uint filterChannel = filterPos.z - (filterIndex * imageDepth);
    uint outputSquare = oDims->width * oDims->height;

    float s = 0;

    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        int inputOffset = (batchIndex * imageDepth + filterChannel) * imageSquare;
        int outputOffset = (batchIndex * filtersCount + filterIndex) * outputSquare;

        for (int y = 0; y < oDims->height; ++y) {
            int iy = y * int(stride) - int(padding) + int(filterPos.y);
            int oy = y;
            if (iy < 0 || iy >= iDims->height) {
                continue;
            }
            for (int x = 0; x < oDims->width; ++x) {
                int ix = x * int(stride) - int(padding) + int(filterPos.x);
                int ox = x;
                if (ix > -1 && ix < iDims->width) {
                    float input = inputData[inputOffset + iy*iDims->width + ix];
                    float gradient = outputGrad[outputOffset + oy*oDims->width + ox];
                    s += input * gradient;
                }
            }
        }
    }

    weightsGrad[filterPos.z * wDims->width * wDims->height + filterPos.y*wDims->width + filterPos.x] += s;
}

kernel void calcBiasGrads(
    device float *biasGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant MTLSize* oDims [[ buffer(2) ]],
    constant uint& filtersCount [[ buffer(3) ]],
    constant uint& batchSize [[ buffer(4) ]],
    const uint filterIndex [[ thread_position_in_grid ]] )
{
    float s = 0;
    uint outputSquare = oDims->width * oDims->height;

    for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex) {
        uint outputOffset = (batchIndex * filtersCount + filterIndex) * outputSquare;

        for (int i = 0; i < outputSquare; ++i) {
            s += outputGrad[outputOffset + i];
        }
    }
    biasGrad[filterIndex] = s;
}
