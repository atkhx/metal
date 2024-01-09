#include <metal_stdlib>

using namespace metal;

kernel void silu(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float val = inputData[id];
    outputData[id] = val / (1.0 + exp(-val));
}

kernel void siluGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    device float *outputGrad [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float val = inputData[id];
    float siluDerivative = outputData[id] + (exp(-val) / pow(1.0 + exp(-val), 2.0));

    inputGrad[id] += outputGrad[id] * siluDerivative;
}
