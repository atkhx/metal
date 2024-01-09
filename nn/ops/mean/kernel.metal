#include <metal_stdlib>

using namespace metal;

kernel void meanByRows(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& chunkSize [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    float sumValue = 0.0;
    for (uint i = gid.y * chunkSize; i < (gid.y+1) * chunkSize; ++i) {
        sumValue += inputData[i];
    }
    outputData[gid.y] = sumValue/float(chunkSize);
}

kernel void meanByRowsGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& chunkSize [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    inputGrad[id] += outputGrad[id/chunkSize];
}
