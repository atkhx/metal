#include <metal_stdlib>

using namespace metal;

kernel void dropout(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    device float *randomData [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (randomData[id] > probability) {
        outputData[id] = inputData[id];
    } else {
        outputData[id] = 0.0;
    }
}

kernel void dropoutGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device float *randomData [[ buffer(2) ]],
    constant float& probability [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    if (randomData[id] > probability) {
        inputGrad[id] += outputGrad[id];
    }
}
