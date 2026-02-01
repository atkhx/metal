#include <metal_stdlib>

using namespace metal;

kernel void positionalAdd(
    device float *inputData [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    constant uint& colsCount [[ buffer(3) ]],
    constant uint& rowsCount [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint idx = gid.z * rowsCount * colsCount + gid.y * colsCount + gid.x;
    uint widx = gid.y * colsCount + gid.x;
    outputData[idx] = inputData[idx] + weightsData[widx];
}

kernel void positionalAddGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& colsCount [[ buffer(2) ]],
    constant uint& rowsCount [[ buffer(3) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint idx = gid.z * rowsCount * colsCount + gid.y * colsCount + gid.x;
    inputGrad[idx] += outputGrad[idx];
}

kernel void positionalAddWeightsGrads(
    device float *weightsGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& colsCount [[ buffer(2) ]],
    constant uint& rowsCount [[ buffer(3) ]],
    constant uint& depth [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint widx = gid.y * colsCount + gid.x;
    float val = 0.0;
    for (uint z = 0; z < depth; ++z) {
        uint idx = z * rowsCount * colsCount + widx;
        val += outputGrad[idx];
    }
    weightsGrad[widx] += val;
}
