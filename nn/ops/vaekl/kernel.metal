#include <metal_stdlib>

using namespace metal;

kernel void vaeKLForward(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant uint& inW [[ buffer(2) ]],
    constant uint& inH [[ buffer(3) ]],
    constant uint& outW [[ buffer(4) ]],
    constant uint& outH [[ buffer(5) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;
    uint inBase = gid.z * inH * inW + gid.y * inW;

    float mu = inputData[inBase + gid.x];
    float logvar = inputData[inBase + gid.x + outW];

    outputData[outIdx] = 0.5 * (mu * mu + exp(logvar) - logvar - 1.0);
}

kernel void vaeKLBackward(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    constant uint& inW [[ buffer(3) ]],
    constant uint& inH [[ buffer(4) ]],
    constant uint& outW [[ buffer(5) ]],
    constant uint& outH [[ buffer(6) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;
    uint inBase = gid.z * inH * inW + gid.y * inW;

    uint muIdx = inBase + gid.x;
    uint logvarIdx = inBase + gid.x + outW;

    float grad = outputGrad[outIdx];
    float mu = inputData[muIdx];
    float logvar = inputData[logvarIdx];

    inputGrad[muIdx] += grad * mu;
    inputGrad[logvarIdx] += grad * 0.5 * (exp(logvar) - 1.0);
}
