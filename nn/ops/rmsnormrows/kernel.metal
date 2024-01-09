#include <metal_stdlib>

using namespace metal;

kernel void calcRMSByRows(
    device float *input [[ buffer(0) ]],
    device float *rmsData [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint i = gid.y*width; i < (gid.y+1)*width; ++i) {
        val += input[i] * input[i];
    }
    rmsData[gid.y] = sqrt(1e-5 + (val / float(width)));
}

kernel void normByRMS(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    device float *rmsData [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    output[gid.y*width + gid.x] = input[gid.y*width + gid.x] / rmsData[gid.y];
}

kernel void calcRMSGrads(
    device float *rmsData [[ buffer(0) ]],
    device float *rmsGrad [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    device float *outputGrad [[ buffer(3) ]],
    constant uint& chunkSize [[ buffer(4) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float val = 0.0;
    for (uint i = id * chunkSize; i < (id+1)*chunkSize; ++i) {
        val -= outputGrad[i] * outputData[i];
    }
    rmsGrad[id] = val / (rmsData[id] * rmsData[id] * float(chunkSize));
}

kernel void calcInputGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    device float *rmsData [[ buffer(3) ]],
    device float *rmsGrad [[ buffer(4) ]],
    constant uint& chunkSize [[ buffer(5) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    uint row = id/chunkSize;

    inputGrad[id] += outputGrad[id]/rmsData[row] + (rmsGrad[row] * inputData[id]);
}

