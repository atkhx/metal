#include <metal_stdlib>

using namespace metal;

kernel void sanitize(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float v = inputData[id];
    if (isnan(v) || isinf(v)) {
        v = 0.0;
    }
    outputData[id] = v;
}

kernel void sanitizeGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float v = inputData[id];
    if (!(isnan(v) || isinf(v))) {
        inputGrad[id] += outputGrad[id];
    }
}
