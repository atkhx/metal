#include <metal_stdlib>

using namespace metal;

kernel void nllByPos(
    device float *softmax [[ buffer(0) ]],
    device float *output [[ buffer(1) ]],
    device float *targets [[ buffer(2) ]],
    constant uint& width [[ buffer(3) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    output[id] = -log(softmax[id * width + int(targets[id])]);
}

kernel void nllByPosBwd(
    device float *outputData [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device float *targets [[ buffer(2) ]],
    device float *softmax [[ buffer(3) ]],
    device float *nllGrad [[ buffer(4) ]],
    constant uint& chunkSize [[ buffer(5) ]],
    const uint id [[ thread_position_in_grid ]] )
{
    uint rowIdx = id / chunkSize;
    uint tgtIndex = targets[rowIdx];

    if (id == tgtIndex + (rowIdx * chunkSize)) {
        nllGrad[id] = outputGrad[rowIdx] * (-1.0 / softmax[id]);
    } else {
        nllGrad[id] = 0;
    }
}
