#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation MulRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _mulRowsPSO;
    id<MTLComputePipelineState> _calcInputGradsPSO;
    id<MTLComputePipelineState> _calcWeightsGradsPSO;

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

        _mulRowsPSO = [self createPipelineStateWithFunctionName:@"mulRows"];
        _calcInputGradsPSO   = [self createPipelineStateWithFunctionName:@"calcInputGrads"];
        _calcWeightsGradsPSO = [self createPipelineStateWithFunctionName:@"calcWeightsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> mulRows = [commandBuffer computeCommandEncoder];
    [mulRows setComputePipelineState:_mulRowsPSO];
    [mulRows setBuffer:inputData offset:0 atIndex:0];
    [mulRows setBuffer:weightsData offset:0 atIndex:1];
    [mulRows setBuffer:outputData offset:0 atIndex:2];
    [mulRows setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [mulRows dispatchThreads:MTLSizeMake(chunkSize, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [mulRows endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_calcInputGradsPSO];
    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:0];
    [calcInputGrads setBuffer:weightsData offset:0 atIndex:1];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:2];
    [calcInputGrads setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [calcInputGrads dispatchThreads:MTLSizeMake(chunkSize, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcInputGrads endEncoding];

    id<MTLComputeCommandEncoder> calcWeightsGrads = [commandBuffer computeCommandEncoder];
    [calcWeightsGrads setComputePipelineState:_calcWeightsGradsPSO];
    [calcWeightsGrads setBuffer:inputData offset:0 atIndex:0];
    [calcWeightsGrads setBuffer:weightsGrad offset:0 atIndex:1];
    [calcWeightsGrads setBuffer:outputGrad offset:0 atIndex:2];
    [calcWeightsGrads setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [calcWeightsGrads setBytes:&rowsCount length:sizeof(uint) atIndex:4];
    [calcWeightsGrads dispatchThreads:MTLSizeMake(chunkSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcWeightsGrads endEncoding];
}

@end