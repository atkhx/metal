#include <metal_stdlib>

using namespace metal;

constant float kSqrt2OverPi = 0.7978845608028654;
constant float kGeluC = 0.044715;

kernel void gelunew(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float x = inputData[id];
    float x3 = x * x * x;
    float u = kSqrt2OverPi * (x + kGeluC * x3);
    float t = tanh(u);
    outputData[id] = 0.5 * x * (1.0 + t);
}

kernel void gelunewGrads(
    device float *inputData [[ buffer(0) ]],
    device float *inputGrad [[ buffer(1) ]],
    device float *outputData [[ buffer(2) ]],
    device float *outputGrad [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    float x = inputData[id];
    float x2 = x * x;
    float x3 = x2 * x;
    float u = kSqrt2OverPi * (x + kGeluC * x3);
    float t = tanh(u);
    float du = kSqrt2OverPi * (1.0 + 3.0 * kGeluC * x2);
    float dt = (1.0 - t * t) * du;
    float geluDeriv = 0.5 * (1.0 + t) + 0.5 * x * dt;
    inputGrad[id] += outputGrad[id] * geluDeriv;
}
