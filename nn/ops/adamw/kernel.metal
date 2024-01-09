#include <metal_stdlib>

using namespace metal;

kernel void updateWithAdam(
    device float *dataBuffer [[ buffer(0) ]],
    device float *gradBuffer [[ buffer(1) ]],
    device float *mBuffer [[ buffer(2) ]],
    device float *vBuffer [[ buffer(3) ]],
    constant float& beta1 [[ buffer(4) ]],
    constant float& beta2 [[ buffer(5) ]],
    constant float& beta1powIterationLR [[ buffer(6) ]],
    constant float& beta2powIteration [[ buffer(7) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    mBuffer[id] = beta1*mBuffer[id] + (1 - beta1)*gradBuffer[id];
    vBuffer[id] = beta2*vBuffer[id] + (1 - beta2)*gradBuffer[id]*gradBuffer[id];

    dataBuffer[id] -= mBuffer[id] * beta1powIterationLR / (sqrt(vBuffer[id] * beta2powIteration) + 0.000000001);
}