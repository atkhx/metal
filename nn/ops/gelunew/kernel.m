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

@implementation GeLuNewKernelImpl {
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

        _geluPSO = [self createPipelineStateWithFunctionName:@"gelunew"];
        _geluGradsPSO = [self createPipelineStateWithFunctionName:@"gelunewGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_geluPSO];
    [encoder setBuffer:inputData offset:0 atIndex:0];
    [encoder setBuffer:outputData offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(inputData.length/sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_geluPSO)];
    [encoder endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:_geluGradsPSO];
    [encoder setBuffer:inputData offset:0 atIndex:0];
    [encoder setBuffer:inputGrad offset:0 atIndex:1];
    [encoder setBuffer:outputData offset:0 atIndex:2];
    [encoder setBuffer:outputGrad offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(inputGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_geluGradsPSO)];
    [encoder endEncoding];
}

@end
