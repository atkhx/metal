#include <metal_stdlib>

using namespace metal;

kernel void maxPoolForward(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    device uint *maskData [[ buffer(2) ]],
    constant uint& inW [[ buffer(3) ]],
    constant uint& inH [[ buffer(4) ]],
    constant uint& outW [[ buffer(5) ]],
    constant uint& outH [[ buffer(6) ]],
    constant uint& poolSize [[ buffer(7) ]],
    constant uint& stride [[ buffer(8) ]],
    constant uint& padding [[ buffer(9) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;

    int inX0 = int(gid.x) * int(stride) - int(padding);
    int inY0 = int(gid.y) * int(stride) - int(padding);

    float maxVal = -INFINITY;
    uint maxIdx = 0xFFFFFFFFu;

    uint base = gid.z * inH * inW;

    for (uint ky = 0; ky < poolSize; ++ky) {
        int iy = inY0 + int(ky);
        if (iy < 0 || iy >= int(inH)) {
            continue;
        }
        uint row = base + uint(iy) * inW;
        for (uint kx = 0; kx < poolSize; ++kx) {
            int ix = inX0 + int(kx);
            if (ix < 0 || ix >= int(inW)) {
                continue;
            }
            uint idx = row + uint(ix);
            float v = inputData[idx];
            if (v > maxVal) {
                maxVal = v;
                maxIdx = idx;
            }
        }
    }

    outputData[outIdx] = maxVal;
    maskData[outIdx] = maxIdx;
}

kernel void maxPoolBackward(
    device float *inputGrad [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device uint *maskData [[ buffer(2) ]],
    constant uint& inW [[ buffer(3) ]],
    constant uint& inH [[ buffer(4) ]],
    constant uint& outW [[ buffer(5) ]],
    constant uint& outH [[ buffer(6) ]],
    constant uint& poolSize [[ buffer(7) ]],
    constant uint& stride [[ buffer(8) ]],
    constant uint& padding [[ buffer(9) ]],
    const uint3 gid [[ thread_position_in_grid ]] )
{
    uint outIdx = gid.z * outH * outW + gid.y * outW + gid.x;
    uint inIdx = maskData[outIdx];
    if (inIdx == 0xFFFFFFFFu) {
        return;
    }

    device atomic_float* grad = (device atomic_float*)inputGrad;
    atomic_fetch_add_explicit(&grad[inIdx], outputGrad[outIdx], memory_order_relaxed);
}
