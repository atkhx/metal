#include <metal_stdlib>

using namespace metal;

kernel void relu(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (inputData[id] < 0) {
        outputData[id] = 0;
    } else {
        outputData[id] = inputData[id];
    }
}

kernel void reluGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (inputData[id] > 0) {
        inputGrad[id] += outputGrad[id];
    }
}
