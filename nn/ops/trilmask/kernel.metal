#include <metal_stdlib>

using namespace metal;

kernel void trilMask(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    constant float& mask [[ buffer(2) ]],
    constant uint& colsCount [[ buffer(3) ]],
    constant uint& rowsCount [[ buffer(4) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.z*colsCount*rowsCount + gid.y*colsCount + gid.x;
    if (gid.x > gid.y) {
        output[i] = mask;
    } else {
        output[i] = input[i];
    }
}

kernel void trilMaskBwd(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& colsCount [[ buffer(2) ]],
    constant uint& rowsCount [[ buffer(3) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint i = gid.z*colsCount*rowsCount + gid.y*colsCount + gid.x;
    if (gid.x > gid.y) {
        inputGrad[i] = 0;
    } else {
        inputGrad[i] = outputGrad[i];
    }
}
