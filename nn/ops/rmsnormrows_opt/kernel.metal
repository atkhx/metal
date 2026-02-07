#include <metal_stdlib>

using namespace metal;

kernel void calcRMSByRowsOpt(
    device float *input [[ buffer(0) ]],
    device float *rmsData [[ buffer(1) ]],
    constant uint& width [[ buffer(2) ]],
    const uint2 gid [[ thread_position_in_grid ]],
    const uint2 tid [[ thread_position_in_threadgroup ]],
    const uint2 tgs [[ threads_per_threadgroup ]] )
{
    uint row = gid.y;
    threadgroup float sumBuf[256];

    float sum = 0.0;
    uint start = row * width;
    for (uint i = tid.x; i < width; i += tgs.x) {
        float x = input[start + i];
        sum += x * x;
    }
    sumBuf[tid.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs.x / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            sumBuf[tid.x] += sumBuf[tid.x + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid.x == 0) {
        rmsData[row] = sqrt(1e-5 + (sumBuf[0] / float(width)));
    }
}

kernel void normByRMSOpt(
    device float *input [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    device float *rmsData [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    output[gid.y*width + gid.x] = input[gid.y*width + gid.x] / rmsData[gid.y];
}

kernel void calcRMSGradsOpt(
    device float *rmsData [[ buffer(0) ]],
    device float *rmsGrad [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    device float *outputGrad [[ buffer(3) ]],
    constant uint& chunkSize [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]],
    const uint2 tid [[ thread_position_in_threadgroup ]],
    const uint2 tgs [[ threads_per_threadgroup ]] )
{
    uint row = gid.y;
    threadgroup float sumBuf[256];

    float sum = 0.0;
    uint start = row * chunkSize;
    for (uint i = tid.x; i < chunkSize; i += tgs.x) {
        uint idx = start + i;
        sum -= outputGrad[idx] * outputData[idx];
    }
    sumBuf[tid.x] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgs.x / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            sumBuf[tid.x] += sumBuf[tid.x + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid.x == 0) {
        rmsGrad[row] = sumBuf[0] / (rmsData[row] * rmsData[row] * float(chunkSize));
    }
}

kernel void calcInputGradsOpt(
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
