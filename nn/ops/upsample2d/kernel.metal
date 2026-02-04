#include <metal_stdlib>

using namespace metal;

kernel void upSample2DForward(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& inW [[ buffer(2) ]],
    constant uint& inH [[ buffer(3) ]],
    constant uint& outW [[ buffer(4) ]],
    constant uint& outH [[ buffer(5) ]],
    constant uint& scale [[ buffer(6) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint inX = gid.x / scale;
    uint inY = gid.y / scale;
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;
    uint inIdx = gid.z * inH * inW + inY * inW + inX;

    outputData[outIdx] = inputData[inIdx];
}

kernel void upSample2DBackward(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    constant uint& inW [[ buffer(2) ]],
    constant uint& inH [[ buffer(3) ]],
    constant uint& outW [[ buffer(4) ]],
    constant uint& outH [[ buffer(5) ]],
    constant uint& scale [[ buffer(6) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint inX = gid.x / scale;
    uint inY = gid.y / scale;
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;
    uint inIdx = gid.z * inH * inW + inY * inW + inX;

    device atomic_float* grad = (device atomic_float*)inputGrad;
    atomic_fetch_add_explicit(&grad[inIdx], outputGrad[outIdx], memory_order_relaxed);
}
