#include <metal_stdlib>

using namespace metal;

kernel void clamp(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    constant float& minValue [[ buffer(2) ]],
    constant float& maxValue [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float v = inputData[id];
    if (v < minValue) {
        v = minValue;
    }
    if (v > maxValue) {
        v = maxValue;
    }
    outputData[id] = v;
}

kernel void clampGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    constant float& minValue [[ buffer(3) ]],
    constant float& maxValue [[ buffer(4) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float v = inputData[id];
    if (v >= minValue && v <= maxValue) {
        inputGrad[id] += outputGrad[id];
    }
}
