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

@implementation SigmoidKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _sigmoidPSO;
    id<MTLComputePipelineState> _sigmoidGradsPSO;

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

        _sigmoidPSO = [self createPipelineStateWithFunctionName:@"sigmoid"];
        _sigmoidGradsPSO = [self createPipelineStateWithFunctionName:@"sigmoidGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> sigmoid = [commandBuffer computeCommandEncoder];

    [sigmoid setComputePipelineState:_sigmoidPSO];
    [sigmoid setBuffer:inputData offset:0 atIndex:0];
    [sigmoid setBuffer:outputData offset:0 atIndex:1];
    [sigmoid dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_sigmoidPSO)];
    [sigmoid endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> sigmoidGrads = [commandBuffer computeCommandEncoder];

    [sigmoidGrads setComputePipelineState:_sigmoidGradsPSO];
    [sigmoidGrads setBuffer:inputGrad offset:0 atIndex:0];
    [sigmoidGrads setBuffer:outputData offset:0 atIndex:1];
    [sigmoidGrads setBuffer:outputGrad offset:0 atIndex:2];
    [sigmoidGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_sigmoidGradsPSO)];
    [sigmoidGrads endEncoding];
}

@end
