#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation ReluKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _reluPSO;
    id<MTLComputePipelineState> _reluGradsPSO;

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

        _reluPSO = [self createPipelineStateWithFunctionName:@"relu"];
        _reluGradsPSO = [self createPipelineStateWithFunctionName:@"reluGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> relu = [commandBuffer computeCommandEncoder];

    [relu setComputePipelineState:_reluPSO];
    [relu setBuffer:inputData offset:0 atIndex:0];
    [relu setBuffer:outputData offset:0 atIndex:1];
    [relu dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [relu endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> reluGrads = [commandBuffer computeCommandEncoder];

    [reluGrads setComputePipelineState:_reluGradsPSO];
    [reluGrads setBuffer:inputData offset:0 atIndex:0];
    [reluGrads setBuffer:inputGrad offset:0 atIndex:1];
    [reluGrads setBuffer:outputGrad offset:0 atIndex:2];
    [reluGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [reluGrads endEncoding];
}

@end