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

@implementation SanitizeKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _sanitizePSO;
    id<MTLComputePipelineState> _sanitizeGradsPSO;

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

        _sanitizePSO      = [self createPipelineStateWithFunctionName:@"sanitize"];
        _sanitizeGradsPSO = [self createPipelineStateWithFunctionName:@"sanitizeGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
{
    id<MTLComputeCommandEncoder> sanitize = [commandBuffer computeCommandEncoder];

    [sanitize setComputePipelineState:_sanitizePSO];
    [sanitize setBuffer:inputData offset:0 atIndex:0];
    [sanitize setBuffer:outputData offset:0 atIndex:1];

    [sanitize dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_sanitizePSO)];
    [sanitize endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
{
    id<MTLComputeCommandEncoder> sanitizeGrads = [commandBuffer computeCommandEncoder];

    [sanitizeGrads setComputePipelineState:_sanitizeGradsPSO];
    [sanitizeGrads setBuffer:inputData offset:0 atIndex:0];
    [sanitizeGrads setBuffer:inputGrad offset:0 atIndex:1];
    [sanitizeGrads setBuffer:outputGrad offset:0 atIndex:2];

    [sanitizeGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_sanitizeGradsPSO)];
    [sanitizeGrads endEncoding];
}

@end
