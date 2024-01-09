#include <metal_stdlib>

using namespace metal;

kernel void embeddings(
    device float *inputData [[ buffer(0) ]],
    device float *outputData [[ buffer(1) ]],
    device float *tokenEmbedding [[ buffer(2) ]],
    constant uint& featuresCount [[ buffer(3) ]],
    constant uint& contextLength [[ buffer(4) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    int outputOffset = gid.y*featuresCount + gid.x;

    int tokenEmbeddingY      = inputData[gid.y];
    int tokenEmbeddingOffset = (tokenEmbeddingY * featuresCount) + gid.x;

    outputData[outputOffset] = tokenEmbedding[tokenEmbeddingOffset];
}

kernel void embeddingsGrads(
    device float *inputData [[ buffer(0) ]],
    device float *outputGrad [[ buffer(1) ]],
    device float *tokenEmbeddingGrad [[ buffer(2) ]],
    constant uint& featuresCount [[ buffer(3) ]],
    const uint2 gid [[ thread_position_in_grid ]] )
{
    int outputOffset = gid.y*featuresCount + gid.x;

    int tokenEmbeddingY      = inputData[gid.y];
    int tokenEmbeddingOffset = (tokenEmbeddingY * featuresCount) + gid.x;

    tokenEmbeddingGrad[tokenEmbeddingOffset] += outputGrad[outputOffset];
}
