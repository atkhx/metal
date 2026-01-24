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
    constant float& eps [[ buffer(8) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float g = gradBuffer[id];
    if (isnan(g) || isinf(g)) {
        g = 0.0;
    }
    mBuffer[id] = beta1*mBuffer[id] + (1 - beta1)*g;
    vBuffer[id] = beta2*vBuffer[id] + (1 - beta2)*g*g;

    dataBuffer[id] -= mBuffer[id] * beta1powIterationLR / (sqrt(vBuffer[id] * beta2powIteration) + eps);
}
