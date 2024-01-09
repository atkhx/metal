#include <metal_stdlib>

using namespace metal;

kernel void transpose(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    constant uint& height [[ buffer(3) ]],
    constant uint& square [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    outputData[gid.z*square + gid.x*height + gid.y] = inputData[gid.z*square + gid.y*width + gid.x];
}

kernel void transposeGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    constant uint& height [[ buffer(3) ]],
    constant uint& square [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    inputGrad[gid.z*square + gid.y*width + gid.x] += outputGrad[gid.z*square + gid.x*height + gid.y];
}
