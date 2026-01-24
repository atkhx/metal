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

@implementation ClampKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _clampPSO;
    id<MTLComputePipelineState> _clampGradsPSO;

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

        _clampPSO      = [self createPipelineStateWithFunctionName:@"clamp"];
        _clampGradsPSO = [self createPipelineStateWithFunctionName:@"clampGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        minValue:(float)minValue
        maxValue:(float)maxValue
{
    id<MTLComputeCommandEncoder> clamp = [commandBuffer computeCommandEncoder];

    [clamp setComputePipelineState:_clampPSO];
    [clamp setBuffer:inputData offset:0 atIndex:0];
    [clamp setBuffer:outputData offset:0 atIndex:1];
    [clamp setBytes:&minValue length:sizeof(float) atIndex:2];
    [clamp setBytes:&maxValue length:sizeof(float) atIndex:3];

    [clamp dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_clampPSO)];
    [clamp endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        minValue:(float)minValue
        maxValue:(float)maxValue
{
    id<MTLComputeCommandEncoder> clampGrads = [commandBuffer computeCommandEncoder];

    [clampGrads setComputePipelineState:_clampGradsPSO];
    [clampGrads setBuffer:inputData offset:0 atIndex:0];
    [clampGrads setBuffer:inputGrad offset:0 atIndex:1];
    [clampGrads setBuffer:outputGrad offset:0 atIndex:2];
    [clampGrads setBytes:&minValue length:sizeof(float) atIndex:3];
    [clampGrads setBytes:&maxValue length:sizeof(float) atIndex:4];

    [clampGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_clampGradsPSO)];
    [clampGrads endEncoding];
}

@end
