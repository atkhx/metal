#include <metal_stdlib>

using namespace metal;

kernel void addEqual(
    device float *inputData [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    outputData[id] = inputData[id] + weightsData[id];
}

kernel void calcGrads(
    device float *inputGrad [[ buffer(0) ]],
    device float *weightGrad [[ buffer(1) ]],
    device float *outputGrad [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    inputGrad[id] += outputGrad[id];
    weightGrad[id] += outputGrad[id];
}

