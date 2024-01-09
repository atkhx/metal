#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation NegLogLikelihoodKernelImpl {
    id<MTLDevice> _device;

    // negative log likelihood
    id<MTLComputePipelineState> _nllByPosPSO;
    id<MTLComputePipelineState> _nllByPosBwdPOS;

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

        _nllByPosPSO = [self createPipelineStateWithFunctionName:@"nllByPos"];
        _nllByPosBwdPOS = [self createPipelineStateWithFunctionName:@"nllByPosBwd"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        softmax:(id<MTLBuffer>)softmax
        output:(id<MTLBuffer>)output
        targets:(id<MTLBuffer>)targets
        chunkSize:(uint)chunkSize
{
    uint rowsCount = targets.length / sizeof(float);

    id<MTLComputeCommandEncoder> negativeLogLikelihood = [commandBuffer computeCommandEncoder];
    [negativeLogLikelihood setComputePipelineState:_nllByPosPSO];
    [negativeLogLikelihood setBuffer:softmax offset:0 atIndex:0];
    [negativeLogLikelihood setBuffer:output offset:0 atIndex:1];
    [negativeLogLikelihood setBuffer:targets offset:0 atIndex:2];
    [negativeLogLikelihood setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [negativeLogLikelihood dispatchThreads:MTLSizeMake(rowsCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [negativeLogLikelihood endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        targets:(id<MTLBuffer>)targets
        softmax:(id<MTLBuffer>)softmax
        nllGrad:(id<MTLBuffer>)nllGrad
        chunkSize:(uint)chunkSize;
{
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:_nllByPosBwdPOS];
    [computeEncoder setBuffer:outputData offset:0 atIndex:0];
    [computeEncoder setBuffer:outputGrad offset:0 atIndex:1];
    [computeEncoder setBuffer:targets offset:0 atIndex:2];
    [computeEncoder setBuffer:softmax offset:0 atIndex:3];
    [computeEncoder setBuffer:nllGrad offset:0 atIndex:4];
    [computeEncoder setBytes:&chunkSize length:sizeof(uint) atIndex:5];
    [computeEncoder dispatchThreads:MTLSizeMake(nllGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [computeEncoder endEncoding];
}


@end