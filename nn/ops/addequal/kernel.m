#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation addEqualKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _addEqualPSO;
    id<MTLComputePipelineState> _calcGradsPSO;

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

        _addEqualPSO  = [self createPipelineStateWithFunctionName:@"addEqual"];
        _calcGradsPSO = [self createPipelineStateWithFunctionName:@"calcGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> addEqual = [commandBuffer computeCommandEncoder];
    [addEqual setComputePipelineState:_addEqualPSO];
    [addEqual setBuffer:inputData offset:0 atIndex:0];
    [addEqual setBuffer:weightsData offset:0 atIndex:1];
    [addEqual setBuffer:outputData offset:0 atIndex:2];
    [addEqual dispatchThreads:MTLSizeMake(inputData.length/sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [addEqual endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> calcGrads = [commandBuffer computeCommandEncoder];
    [calcGrads setComputePipelineState:_calcGradsPSO];
    [calcGrads setBuffer:inputGrad offset:0 atIndex:0];
    [calcGrads setBuffer:weightsGrad offset:0 atIndex:1];
    [calcGrads setBuffer:outputGrad offset:0 atIndex:2];
    [calcGrads dispatchThreads:MTLSizeMake(inputGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcGrads endEncoding];
}

@end