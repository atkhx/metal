#include <metal_stdlib>

using namespace metal;

kernel void calcMeanInvStdOpt(
    device float *input [[ buffer(0) ]],
    device float *meanData [[ buffer(1) ]],
    device float *invStdData [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    constant float& eps [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]],
    const uint2 tid [[ thread_position_in_threadgroup ]],
    const uint2 tgs [[ threads_per_threadgroup ]] )
{
    uint row = gid.y;
    threadgroup float sumBuf[256];
    threadgroup float sumSqBuf[256];

    float sum = 0.0;
    float sumSq = 0.0;
    uint start = row * width;
    for (uint i = tid.x; i < width; i += tgs.x) {
        float x = input[start + i];
        sum += x;
        sumSq += x * x;
    }
    sumBuf[tid.x] = sum;
    sumSqBuf[tid.x] = sumSq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs.x / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            sumBuf[tid.x] += sumBuf[tid.x + s];
            sumSqBuf[tid.x] += sumSqBuf[tid.x + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid.x == 0) {
        float invW = 1.0 / float(width);
        float mean = sumBuf[0] * invW;
        float var = sumSqBuf[0] * invW - mean * mean;
        meanData[row] = mean;
        invStdData[row] = rsqrt(var + eps);
    }
}

kernel void normByStatsOpt(
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

kernel void calcRowSumsOpt(
    device float *inputData [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device float *meanData [[ buffer(2) ]],
    device float *sumDy [[ buffer(3) ]],
    device float *sumDyXmu [[ buffer(4) ]],
    constant uint& width [[ buffer(5) ]],
    const uint2 gid [[ thread_position_in_grid ]],
    const uint2 tid [[ thread_position_in_threadgroup ]],
    const uint2 tgs [[ threads_per_threadgroup ]] )
{
    uint row = gid.y;
    threadgroup float sumBuf[256];
    threadgroup float sumBuf2[256];

    float s1 = 0.0;
    float s2 = 0.0;
    uint start = row * width;
    float mean = meanData[row];
    for (uint i = tid.x; i < width; i += tgs.x) {
        float dy = outputGrad[start + i];
        float xmu = inputData[start + i] - mean;
        s1 += dy;
        s2 += dy * xmu;
    }
    sumBuf[tid.x] = s1;
    sumBuf2[tid.x] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs.x / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            sumBuf[tid.x] += sumBuf[tid.x + s];
            sumBuf2[tid.x] += sumBuf2[tid.x + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid.x == 0) {
        sumDy[row] = sumBuf[0];
        sumDyXmu[row] = sumBuf2[0];
    }
}

kernel void calcInputGradsOpt(
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
