#include <metal_stdlib>

using namespace metal;

kernel void vaeSampleForward(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    device float *epsData [[ buffer(2) ]],
    device float *randomData [[ buffer(3) ]],
    constant uint& inW [[ buffer(4) ]],
    constant uint& inH [[ buffer(5) ]],
    constant uint& outW [[ buffer(6) ]],
    constant uint& outH [[ buffer(7) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;
    uint inBase = gid.z * inH * inW + gid.y * inW;

    uint randBase = (gid.z * outH + gid.y) * (outW * 2);
    uint randIdx = randBase + gid.x * 2;
    float u1 = max(randomData[randIdx], 1.0e-7);
    float u2 = randomData[randIdx + 1];

    float r = sqrt(-2.0 * log(u1));
    float theta = 6.28318530718 * u2;
    float eps = r * cos(theta);

    float mu = inputData[inBase + gid.x];
    float logvar = inputData[inBase + gid.x + outW];
    float sigma = exp(0.5 * logvar);

    outputData[outIdx] = mu + sigma * eps;
    epsData[outIdx] = eps;
}

kernel void vaeSampleBackward(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    device float *epsData [[ buffer(3) ]],
    constant uint& inW [[ buffer(4) ]],
    constant uint& inH [[ buffer(5) ]],
    constant uint& outW [[ buffer(6) ]],
    constant uint& outH [[ buffer(7) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;
    uint inBase = gid.z * inH * inW + gid.y * inW;

    uint muIdx = inBase + gid.x;
    uint logvarIdx = inBase + gid.x + outW;

    float grad = outputGrad[outIdx];
    inputGrad[muIdx] += grad;

    float logvar = inputData[logvarIdx];
    float sigma = exp(0.5 * logvar);
    inputGrad[logvarIdx] += grad * 0.5 * sigma * epsData[outIdx];
}
