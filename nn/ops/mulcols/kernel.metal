#include <metal_stdlib>

using namespace metal;

kernel void mulCols(
    device float *inputData [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    constant uint& rowWidth [[ buffer(3) ]],
    constant uint& colHeight [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.z*rowWidth*colHeight + gid.y*rowWidth + gid.x;

    outputData[i] = inputData[i] * weightsData[gid.y];
}

kernel void calcInputGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    constant uint& rowWidth [[ buffer(3) ]],
    constant uint& colHeight [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.z*rowWidth*colHeight + gid.y*rowWidth + gid.x;
    inputGrad[i] += outputGrad[i] * weightsData[gid.y];
}

kernel void calcWeightsGrads(
    device float *inputData [[ buffer(0) ]],
    device float *weightsGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    constant uint& colsCount [[ buffer(3) ]],
    constant uint& rowsCount [[ buffer(4) ]],
    constant uint& depth [[ buffer(5) ]],
    const uint row [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint z = 0; z < depth; ++z) {
        for (uint col = 0; col < colsCount; ++col) {
            val += inputData[z*colsCount*rowsCount + row*colsCount + col] * outputGrad[z*colsCount*rowsCount + row*colsCount + col];
        }
    }
    weightsGrad[row] += val;
}

