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

@implementation LayerNormRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _calcMeanInvStdPSO;
    id<MTLComputePipelineState> _normByStatsPSO;
    id<MTLComputePipelineState> _calcRowSumsPSO;
    id<MTLComputePipelineState> _calcInputGradsPSO;

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

        _calcMeanInvStdPSO = [self createPipelineStateWithFunctionName:@"calcMeanInvStd"];
        _normByStatsPSO = [self createPipelineStateWithFunctionName:@"normByStats"];
        _calcRowSumsPSO = [self createPipelineStateWithFunctionName:@"calcRowSums"];
        _calcInputGradsPSO = [self createPipelineStateWithFunctionName:@"calcInputGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        meanData:(id<MTLBuffer>)meanData
        invStdData:(id<MTLBuffer>)invStdData
        width:(uint)width
        rowsCount:(uint)rowsCount
{
    id<MTLComputeCommandEncoder> calcMeanInvStd = [commandBuffer computeCommandEncoder];
    [calcMeanInvStd setComputePipelineState:_calcMeanInvStdPSO];
    [calcMeanInvStd setBuffer:inputData offset:0 atIndex:0];
    [calcMeanInvStd setBuffer:meanData offset:0 atIndex:1];
    [calcMeanInvStd setBuffer:invStdData offset:0 atIndex:2];
    [calcMeanInvStd setBytes:&width length:sizeof(uint) atIndex:3];
    [calcMeanInvStd dispatchThreads:MTLSizeMake(rowsCount, 1, 1) threadsPerThreadgroup:threadgroupSize1D(_calcMeanInvStdPSO)];
    [calcMeanInvStd endEncoding];

    id<MTLComputeCommandEncoder> normByStats = [commandBuffer computeCommandEncoder];
    [normByStats setComputePipelineState:_normByStatsPSO];
    [normByStats setBuffer:inputData offset:0 atIndex:0];
    [normByStats setBuffer:outputData offset:0 atIndex:1];
    [normByStats setBuffer:meanData offset:0 atIndex:2];
    [normByStats setBuffer:invStdData offset:0 atIndex:3];
    [normByStats setBytes:&width length:sizeof(uint) atIndex:4];
    [normByStats dispatchThreads:MTLSizeMake(width, rowsCount, 1) threadsPerThreadgroup:threadgroupSize2D(_normByStatsPSO)];
    [normByStats endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        meanData:(id<MTLBuffer>)meanData
        invStdData:(id<MTLBuffer>)invStdData
        sumDy:(id<MTLBuffer>)sumDy
        sumDyXmu:(id<MTLBuffer>)sumDyXmu
        width:(uint)width
        rowsCount:(uint)rowsCount
{
    id<MTLComputeCommandEncoder> calcRowSums = [commandBuffer computeCommandEncoder];
    [calcRowSums setComputePipelineState:_calcRowSumsPSO];
    [calcRowSums setBuffer:inputData offset:0 atIndex:0];
    [calcRowSums setBuffer:outputGrad offset:0 atIndex:1];
    [calcRowSums setBuffer:meanData offset:0 atIndex:2];
    [calcRowSums setBuffer:sumDy offset:0 atIndex:3];
    [calcRowSums setBuffer:sumDyXmu offset:0 atIndex:4];
    [calcRowSums setBytes:&width length:sizeof(uint) atIndex:5];
    [calcRowSums dispatchThreads:MTLSizeMake(rowsCount, 1, 1) threadsPerThreadgroup:threadgroupSize1D(_calcRowSumsPSO)];
    [calcRowSums endEncoding];

    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_calcInputGradsPSO];
    [calcInputGrads setBuffer:inputData offset:0 atIndex:0];
    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:1];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:2];
    [calcInputGrads setBuffer:meanData offset:0 atIndex:3];
    [calcInputGrads setBuffer:invStdData offset:0 atIndex:4];
    [calcInputGrads setBuffer:sumDy offset:0 atIndex:5];
    [calcInputGrads setBuffer:sumDyXmu offset:0 atIndex:6];
    [calcInputGrads setBytes:&width length:sizeof(uint) atIndex:7];
    [calcInputGrads dispatchThreads:MTLSizeMake(inputGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:threadgroupSize1D(_calcInputGradsPSO)];
    [calcInputGrads endEncoding];
}

@end
