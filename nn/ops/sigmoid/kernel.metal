#include <metal_stdlib>

using namespace metal;

kernel void sigmoid(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float val = inputData[id];
    outputData[id] = 1.0 / (1.0 + exp(-val));
}

kernel void sigmoidGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float s = outputData[id];
    inputGrad[id] += outputGrad[id] * s * (1.0 - s);
}
