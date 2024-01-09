#include <metal_stdlib>

using namespace metal;

kernel void mulEqual(
    device float *inputData [[ buffer(0) ]],
    device float *weightsData [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    outputData[id] = inputData[id] * weightsData[id];
}

kernel void calcGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *weightsData [[ buffer(2) ]],
    device float *weightsGrad [[ buffer(3) ]],
    device float *outputGrad [[ buffer(4) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    inputGrad[id] += outputGrad[id] * weightsData[id];
    weightsGrad[id] += outputGrad[id] * inputData[id];
}

