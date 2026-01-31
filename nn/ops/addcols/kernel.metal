#include <metal_stdlib>

using namespace metal;

kernel void addCols(
    device float *inputData [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    constant uint& colsCount [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    outputData[gid.y*colsCount + gid.x] = inputData[gid.y*colsCount + gid.x] + weightsData[gid.y];
}

kernel void calcInputGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    inputGrad[id] += outputGrad[id];
}

kernel void calcWeightsGrads(
    device float *weightsGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& colsCount [[ buffer(2) ]],
    constant uint& rowsCount [[ buffer(3) ]],
    const uint row [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint col = 0; col < colsCount; ++col) {
        val += outputGrad[row*colsCount+col];
    }
    weightsGrad[row] += val;
}
