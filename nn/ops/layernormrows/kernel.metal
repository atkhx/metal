#include <metal_stdlib>

using namespace metal;

constant float kEps = 1e-5;

kernel void calcMeanInvStd(
    device float *input [[ buffer(0) ]],
    device float *meanData [[ buffer(1) ]],
    device float *invStdData [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint row [[ thread_position_in_grid ]] )
{
    float sum = 0.0;
    float sumsq = 0.0;
    uint start = row * width;
    for (uint i = start; i < start + width; ++i) {
        float v = input[i];
        sum += v;
        sumsq += v * v;
    }
    float mean = sum / float(width);
    float var = (sumsq / float(width)) - (mean * mean);
    meanData[row] = mean;
    invStdData[row] = rsqrt(var + kEps);
}

kernel void normByStats(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    device float *meanData [[ buffer(2) ]],
    device float *invStdData [[ buffer(3) ]],
    constant uint& width [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    uint idx = gid.y * width + gid.x;
    float mean = meanData[gid.y];
    float invStd = invStdData[gid.y];
    output[idx] = (input[idx] - mean) * invStd;
}

kernel void calcRowSums(
    device float *inputData [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device float *meanData [[ buffer(2) ]],
    device float *sumDy [[ buffer(3) ]],
    device float *sumDyXmu [[ buffer(4) ]],
    constant uint& width [[ buffer(5) ]],
    const uint row [[ thread_position_in_grid ]] )
{
    float s1 = 0.0;
    float s2 = 0.0;
    uint start = row * width;
    float mean = meanData[row];
    for (uint i = start; i < start + width; ++i) {
        float dy = outputGrad[i];
        float xmu = inputData[i] - mean;
        s1 += dy;
        s2 += dy * xmu;
    }
    sumDy[row] = s1;
    sumDyXmu[row] = s2;
}

kernel void calcInputGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    device float *meanData [[ buffer(3) ]],
    device float *invStdData [[ buffer(4) ]],
    device float *sumDy [[ buffer(5) ]],
    device float *sumDyXmu [[ buffer(6) ]],
    constant uint& width [[ buffer(7) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    uint row = id / width;
    float mean = meanData[row];
    float invStd = invStdData[row];
    float dy = outputGrad[id];
    float xmu = inputData[id] - mean;
    float s1 = sumDy[row];
    float s2 = sumDyXmu[row];
    float w = float(width);
    float grad = invStd * (dy - s1 / w - xmu * invStd * s2 / w);
    inputGrad[id] += grad;
}
