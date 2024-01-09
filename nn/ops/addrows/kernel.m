#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation AddRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _addRowsPSO;
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

        _addRowsPSO = [self createPipelineStateWithFunctionName:@"addRows"];
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

    id<MTLComputeCommandEncoder> addRows = [commandBuffer computeCommandEncoder];
    [addRows setComputePipelineState:_addRowsPSO];
    [addRows setBuffer:inputData offset:0 atIndex:0];
    [addRows setBuffer:weightsData offset:0 atIndex:1];
    [addRows setBuffer:outputData offset:0 atIndex:2];
    [addRows setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [addRows dispatchThreads:MTLSizeMake(chunkSize, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [addRows endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_calcInputGradsPSO];
    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:0];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:1];
    [calcInputGrads dispatchThreads:MTLSizeMake(inputGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcInputGrads endEncoding];

    uint rowsCount = inputGrad.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> calcWeightsGrads = [commandBuffer computeCommandEncoder];
    [calcWeightsGrads setComputePipelineState:_calcWeightsGradsPSO];
    [calcWeightsGrads setBuffer:weightsGrad offset:0 atIndex:0];
    [calcWeightsGrads setBuffer:outputGrad offset:0 atIndex:1];
    [calcWeightsGrads setBytes:&chunkSize length:sizeof(uint) atIndex:2];
    [calcWeightsGrads setBytes:&rowsCount length:sizeof(uint) atIndex:3];
    [calcWeightsGrads dispatchThreads:MTLSizeMake(chunkSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcWeightsGrads endEncoding];
}

@end