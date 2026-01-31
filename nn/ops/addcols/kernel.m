#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

static inline MTLSize threadgroupSize2D(id<MTLComputePipelineState> pso) {
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger max = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger h = max / w;
    if (h < 1) {
        h = 1;
    }
    return MTLSizeMake(w, h, 1);
}

static inline MTLSize threadgroupSize1D(id<MTLComputePipelineState> pso) {
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger max = pso.maxTotalThreadsPerThreadgroup;
    if (w > max) {
        w = max;
    }
    return MTLSizeMake(w, 1, 1);
}

@implementation AddColsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _addColsPSO;
    id<MTLComputePipelineState> _calcInputGradsPSO;
    id<MTLComputePipelineState> _calcWeightsGradsPSO;

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

        _addColsPSO = [self createPipelineStateWithFunctionName:@"addCols"];
        _calcInputGradsPSO = [self createPipelineStateWithFunctionName:@"calcInputGrads"];
        _calcWeightsGradsPSO = [self createPipelineStateWithFunctionName:@"calcWeightsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    id<MTLComputeCommandEncoder> addCols = [commandBuffer computeCommandEncoder];
    [addCols setComputePipelineState:_addColsPSO];
    [addCols setBuffer:inputData offset:0 atIndex:0];
    [addCols setBuffer:weightsData offset:0 atIndex:1];
    [addCols setBuffer:outputData offset:0 atIndex:2];
    [addCols setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [addCols dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:threadgroupSize2D(_addColsPSO)];
    [addCols endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_calcInputGradsPSO];
    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:0];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:1];
    [calcInputGrads dispatchThreads:MTLSizeMake(inputGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_calcInputGradsPSO)];
    [calcInputGrads endEncoding];

    id<MTLComputeCommandEncoder> calcWeightsGrads = [commandBuffer computeCommandEncoder];
    [calcWeightsGrads setComputePipelineState:_calcWeightsGradsPSO];
    [calcWeightsGrads setBuffer:weightsGrad offset:0 atIndex:0];
    [calcWeightsGrads setBuffer:outputGrad offset:0 atIndex:1];
    [calcWeightsGrads setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [calcWeightsGrads setBytes:&rowsCount length:sizeof(uint) atIndex:3];
    [calcWeightsGrads dispatchThreads:MTLSizeMake(rowsCount, 1, 1) threadsPerThreadgroup:threadgroupSize1D(_calcWeightsGradsPSO)];
    [calcWeightsGrads endEncoding];
}

@end
