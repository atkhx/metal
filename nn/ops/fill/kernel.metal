#include <metal_stdlib>

using namespace metal;

kernel void fill(
    device float *dstBuffer [[ buffer(0) ]],
    constant float& value [[ buffer(1) ]],
    const uint id [[ thread_position_in_grid ]])
{
    dstBuffer[id] = value;
}
