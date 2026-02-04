#include <metal_stdlib>

using namespace metal;

kernel void bceForward(
    device float *inputData [[ buffer(0) ]],
    device float *targetData [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float y = inputData[id];
    float t = targetData[id];
    float y1 = max(y, 1.0e-7);
    float y0 = max(1.0 - y, 1.0e-7);

    outputData[id] = -(t * log(y1) + (1.0 - t) * log(y0));
}

kernel void bceBackward(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *targetData [[ buffer(2) ]],
    device float *outputGrad [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float y = inputData[id];
    float t = targetData[id];
    float y1 = max(y, 1.0e-7);
    float y0 = max(1.0 - y, 1.0e-7);

    float grad = (y - t) / (y1 * y0);
    inputGrad[id] += outputGrad[id] * grad;
}
