#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation EmbeddingsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _embeddingsPSO;
    id<MTLComputePipelineState> _embeddingsGradsPSO;

    NSError *error;
}

- (id<MTLComputePipelineState>)createPipelineStateWithFunctionName:(NSString *)functionName {
    id<MTLFunction> function = [self.library newFunctionWithName:functionName];
    if (!function) {
        printf("Failed to load function %s!\n", [functionName UTF8String]);
        return nil;
    }

    id<MTLComputePipelineState> pipelineState = [_device newComputePipelineStateWithFunction:function error:&error];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to create pipeline state: %s\n", errorCString);
        return nil;
    }
    return pipelineState;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;

        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];

        _embeddingsPSO      = [self createPipelineStateWithFunctionName:@"embeddings"];
        _embeddingsGradsPSO = [self createPipelineStateWithFunctionName:@"embeddingsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength
{
    uint rowsCount = outputData.length/(sizeof(float)*featuresCount);

    id<MTLComputeCommandEncoder> embeddings = [commandBuffer computeCommandEncoder];

    [embeddings setComputePipelineState:_embeddingsPSO];
    [embeddings setBuffer:inputData offset:0 atIndex:0];
    [embeddings setBuffer:outputData offset:0 atIndex:1];
    [embeddings setBuffer:tokenEmbedding offset:0 atIndex:2];
    [embeddings setBytes:&featuresCount length:sizeof(uint) atIndex:3];
    [embeddings setBytes:&contextLength length:sizeof(uint) atIndex:4];
    [embeddings dispatchThreads:MTLSizeMake(featuresCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [embeddings endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount
{
    uint rowsCount = outputGrad.length/(sizeof(float)*featuresCount);

    id<MTLComputeCommandEncoder> embeddingsGrads = [commandBuffer computeCommandEncoder];
    [embeddingsGrads setComputePipelineState:_embeddingsGradsPSO];
    [embeddingsGrads setBuffer:inputData offset:0 atIndex:0];
    [embeddingsGrads setBuffer:outputGrad offset:0 atIndex:1];
    [embeddingsGrads setBuffer:tokenEmbeddingGrad offset:0 atIndex:2];
    [embeddingsGrads setBytes:&featuresCount length:sizeof(uint) atIndex:3];
    [embeddingsGrads dispatchThreads:MTLSizeMake(featuresCount, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [embeddingsGrads endEncoding];
}

@end