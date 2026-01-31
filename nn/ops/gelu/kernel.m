#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

static inline MTLSize threadgroupSize1D(id<MTLComputePipelineState> pso) {
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger max = pso.maxTotalThreadsPerThreadgroup;
    if (w > max) {
        w = max;
    }
    return MTLSizeMake(w, 1, 1);
}

@implementation GeluKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _geluPSO;
    id<MTLComputePipelineState> _geluGradsPSO;

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

        _geluPSO = [self createPipelineStateWithFunctionName:@"gelu"];
        _geluGradsPSO = [self createPipelineStateWithFunctionName:@"geluGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> gelu = [commandBuffer computeCommandEncoder];
    [gelu setComputePipelineState:_geluPSO];
    [gelu setBuffer:inputData offset:0 atIndex:0];
    [gelu setBuffer:outputData offset:0 atIndex:1];
    [gelu dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_geluPSO)];
    [gelu endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> geluGrads = [commandBuffer computeCommandEncoder];
    [geluGrads setComputePipelineState:_geluGradsPSO];
    [geluGrads setBuffer:inputData offset:0 atIndex:0];
    [geluGrads setBuffer:inputGrad offset:0 atIndex:1];
    [geluGrads setBuffer:outputData offset:0 atIndex:2];
    [geluGrads setBuffer:outputGrad offset:0 atIndex:3];
    [geluGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_geluGradsPSO)];
    [geluGrads endEncoding];
}

@end
