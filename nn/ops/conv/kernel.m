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

@implementation convKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _convPSO;
    id<MTLComputePipelineState> _calcInputGradsPSO;
    id<MTLComputePipelineState> _calcWeightGradsPSO;
    id<MTLComputePipelineState> _calcBiasGradsPSO;

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

        _convPSO  = [self createPipelineStateWithFunctionName:@"conv"];
        _calcInputGradsPSO = [self createPipelineStateWithFunctionName:@"calcInputGrads"];
        _calcWeightGradsPSO = [self createPipelineStateWithFunctionName:@"calcWeightGrads"];
        _calcBiasGradsPSO = [self createPipelineStateWithFunctionName:@"calcBiasGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        biasesData:(id<MTLBuffer>)biasesData
        outputData:(id<MTLBuffer>)outputData
        iDims:(MTLSize)iDims
        wDims:(MTLSize)wDims
        oDims:(MTLSize)oDims
        filtersCount:(uint)filtersCount
        batchSize:(uint)batchSize
        padding:(uint)padding
        stride:(uint)stride
{
    id<MTLComputeCommandEncoder> conv = [commandBuffer computeCommandEncoder];
    [conv setComputePipelineState:_convPSO];

    [conv setBuffer:inputData offset:0 atIndex:0];
    [conv setBuffer:weightsData offset:0 atIndex:1];
    [conv setBuffer:biasesData offset:0 atIndex:2];
    [conv setBuffer:outputData offset:0 atIndex:3];

    [conv setBytes:&iDims length:sizeof(MTLSize) atIndex:4];
    [conv setBytes:&wDims length:sizeof(MTLSize) atIndex:5];
    [conv setBytes:&oDims length:sizeof(MTLSize) atIndex:6];

    [conv setBytes:&filtersCount length:sizeof(uint) atIndex:7];
    [conv setBytes:&batchSize length:sizeof(uint) atIndex:8];
    [conv setBytes:&padding length:sizeof(uint) atIndex:9];
    [conv setBytes:&stride length:sizeof(uint) atIndex:10];

    [conv dispatchThreads:MTLSizeMake(oDims.width, oDims.height, oDims.depth) threadsPerThreadgroup:threadgroupSize1D(_convPSO)];
    [conv endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        biasesGrad:(id<MTLBuffer>)biasesGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        iDims:(MTLSize)iDims
        wDims:(MTLSize)wDims
        oDims:(MTLSize)oDims
        filtersCount:(uint)filtersCount
        batchSize:(uint)batchSize
        padding:(uint)padding
        stride:(uint)stride
{
    // Calculate input gradients
    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_calcInputGradsPSO];

    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:0];
    [calcInputGrads setBuffer:weightsData offset:0 atIndex:1];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:2];

    [calcInputGrads setBytes:&iDims length:sizeof(MTLSize) atIndex:3];
    [calcInputGrads setBytes:&wDims length:sizeof(MTLSize) atIndex:4];
    [calcInputGrads setBytes:&oDims length:sizeof(MTLSize) atIndex:5];

    [calcInputGrads setBytes:&filtersCount length:sizeof(uint) atIndex:6];
    [calcInputGrads setBytes:&batchSize length:sizeof(uint) atIndex:7];
    [calcInputGrads setBytes:&padding length:sizeof(uint) atIndex:8];
    [calcInputGrads setBytes:&stride length:sizeof(uint) atIndex:9];

    [calcInputGrads dispatchThreads:MTLSizeMake(iDims.width, iDims.height, iDims.depth) threadsPerThreadgroup:threadgroupSize1D(_calcInputGradsPSO)];
    [calcInputGrads endEncoding];

    // Calculate weight gradients
    id<MTLComputeCommandEncoder> calcWeightGrads = [commandBuffer computeCommandEncoder];
    [calcWeightGrads setComputePipelineState:_calcWeightGradsPSO];

    [calcWeightGrads setBuffer:inputData offset:0 atIndex:0];
    [calcWeightGrads setBuffer:weightsGrad offset:0 atIndex:1];
    [calcWeightGrads setBuffer:outputGrad offset:0 atIndex:2];

    [calcWeightGrads setBytes:&iDims length:sizeof(MTLSize) atIndex:3];
    [calcWeightGrads setBytes:&wDims length:sizeof(MTLSize) atIndex:4];
    [calcWeightGrads setBytes:&oDims length:sizeof(MTLSize) atIndex:5];

    [calcWeightGrads setBytes:&filtersCount length:sizeof(uint) atIndex:6];
    [calcWeightGrads setBytes:&batchSize length:sizeof(uint) atIndex:7];
    [calcWeightGrads setBytes:&padding length:sizeof(uint) atIndex:8];
    [calcWeightGrads setBytes:&stride length:sizeof(uint) atIndex:9];

    [calcWeightGrads dispatchThreads:MTLSizeMake(wDims.width, wDims.height, wDims.depth) threadsPerThreadgroup:threadgroupSize1D(_calcWeightGradsPSO)];
    [calcWeightGrads endEncoding];

    // Calculate bias gradients
    id<MTLComputeCommandEncoder> calcBiasGrads = [commandBuffer computeCommandEncoder];
    [calcBiasGrads setComputePipelineState:_calcBiasGradsPSO];

    [calcBiasGrads setBuffer:biasesGrad offset:0 atIndex:0];
    [calcBiasGrads setBuffer:outputGrad offset:0 atIndex:1];
    [calcBiasGrads setBytes:&oDims length:sizeof(MTLSize) atIndex:2];
    [calcBiasGrads setBytes:&filtersCount length:sizeof(uint) atIndex:3];
    [calcBiasGrads setBytes:&batchSize length:sizeof(uint) atIndex:4];

    // range by filtersCount
    [calcBiasGrads dispatchThreads:MTLSizeMake(oDims.depth/batchSize, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [calcBiasGrads endEncoding];
}

@end
