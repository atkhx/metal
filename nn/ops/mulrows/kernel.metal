#include <metal_stdlib>

using namespace metal;

kernel void mulRows(
    device float *inputData [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    constant uint& chunkSize [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    outputData[gid.y*chunkSize + gid.x] = inputData[gid.y*chunkSize + gid.x] * weightsData[gid.x];
}

kernel void calcInputGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    constant uint& chunkSize [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.y*chunkSize + gid.x;
    inputGrad[i] += outputGrad[i] * weightsData[gid.x];
}

kernel void calcWeightsGrads(
    device float *inputData [[ buffer(0) ]],
    device float *weightsGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    constant uint& colsCount [[ buffer(3) ]],
    constant uint& rowsCount [[ buffer(4) ]],
    const uint col [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint row = 0; row < rowsCount; ++row) {
        val += inputData[row*colsCount+col] * outputGrad[row*colsCount+col];
    }
    weightsGrad[col] += val;
}

