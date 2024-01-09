#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation SiluKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _siluPSO;
    id<MTLComputePipelineState> _siluGradsPSO;

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

        _siluPSO = [self createPipelineStateWithFunctionName:@"silu"];
        _siluGradsPSO = [self createPipelineStateWithFunctionName:@"siluGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> silu = [commandBuffer computeCommandEncoder];

    [silu setComputePipelineState:_siluPSO];
    [silu setBuffer:inputData offset:0 atIndex:0];
    [silu setBuffer:outputData offset:0 atIndex:1];
    [silu dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [silu endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> siluGrads = [commandBuffer computeCommandEncoder];

    [siluGrads setComputePipelineState:_siluGradsPSO];
    [siluGrads setBuffer:inputData offset:0 atIndex:0];
    [siluGrads setBuffer:inputGrad offset:0 atIndex:1];
    [siluGrads setBuffer:outputData offset:0 atIndex:2];
    [siluGrads setBuffer:outputGrad offset:0 atIndex:3];
    [siluGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [siluGrads endEncoding];
}

@end