#include <metal_stdlib>

using namespace metal;

kernel void packCHWToTexture(
    device float *inputData [[ buffer(0) ]],
    texture2d_array<float, access::write> outTex [[ texture(0) ]],
    constant uint& width [[ buffer(1) ]],
    constant uint& height [[ buffer(2) ]],
    constant uint& channels [[ buffer(3) ]],
    constant uint& batchSize [[ buffer(4) ]],
    constant uint& slices [[ buffer(5) ]],
    const uint3 gid [[ thread_position_in_grid ]]
) {
    if (gid.x >= width || gid.y >= height) return;
    if (gid.z >= batchSize * slices) return;

    uint batch = gid.z / slices;
    uint slice = gid.z - batch * slices;
    uint baseChannel = slice * 4;

    uint hw = width * height;
    uint base = batch * channels * hw + gid.y * width + gid.x;

    float4 v = float4(0.0);
    if (baseChannel + 0 < channels) v.x = inputData[base + (baseChannel + 0) * hw];
    if (baseChannel + 1 < channels) v.y = inputData[base + (baseChannel + 1) * hw];
    if (baseChannel + 2 < channels) v.z = inputData[base + (baseChannel + 2) * hw];
    if (baseChannel + 3 < channels) v.w = inputData[base + (baseChannel + 3) * hw];

    outTex.write(v, uint2(gid.x, gid.y), gid.z);
}

kernel void unpackTextureToCHW(
    texture2d_array<float, access::read> inTex [[ texture(0) ]],
    device float *outputData [[ buffer(0) ]],
    constant uint& width [[ buffer(1) ]],
    constant uint& height [[ buffer(2) ]],
    constant uint& channels [[ buffer(3) ]],
    constant uint& batchSize [[ buffer(4) ]],
    constant uint& slices [[ buffer(5) ]],
    const uint3 gid [[ thread_position_in_grid ]]
) {
    if (gid.x >= width || gid.y >= height) return;
    if (gid.z >= batchSize * slices) return;

    uint batch = gid.z / slices;
    uint slice = gid.z - batch * slices;
    uint baseChannel = slice * 4;

    uint hw = width * height;
    uint base = batch * channels * hw + gid.y * width + gid.x;

    float4 v = inTex.read(uint2(gid.x, gid.y), gid.z);
    if (baseChannel + 0 < channels) outputData[base + (baseChannel + 0) * hw] = v.x;
    if (baseChannel + 1 < channels) outputData[base + (baseChannel + 1) * hw] = v.y;
    if (baseChannel + 2 < channels) outputData[base + (baseChannel + 2) * hw] = v.z;
    if (baseChannel + 3 < channels) outputData[base + (baseChannel + 3) * hw] = v.w;
}
