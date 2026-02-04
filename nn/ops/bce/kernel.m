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

@implementation BCEKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _bcePSO;
    id<MTLComputePipelineState> _bceGradsPSO;

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

        _bcePSO = [self createPipelineStateWithFunctionName:@"bceForward"];
        _bceGradsPSO = [self createPipelineStateWithFunctionName:@"bceBackward"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        targetData:(id<MTLBuffer>)targetData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> bce = [commandBuffer computeCommandEncoder];

    [bce setComputePipelineState:_bcePSO];
    [bce setBuffer:inputData offset:0 atIndex:0];
    [bce setBuffer:targetData offset:0 atIndex:1];
    [bce setBuffer:outputData offset:0 atIndex:2];
    [bce dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_bcePSO)];
    [bce endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        targetData:(id<MTLBuffer>)targetData
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> bceGrads = [commandBuffer computeCommandEncoder];

    [bceGrads setComputePipelineState:_bceGradsPSO];
    [bceGrads setBuffer:inputData offset:0 atIndex:0];
    [bceGrads setBuffer:inputGrad offset:0 atIndex:1];
    [bceGrads setBuffer:targetData offset:0 atIndex:2];
    [bceGrads setBuffer:outputGrad offset:0 atIndex:3];
    [bceGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_bceGradsPSO)];
    [bceGrads endEncoding];
}

@end
