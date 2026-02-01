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

@implementation PositionalAddKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _positionalAddPSO;
    id<MTLComputePipelineState> _positionalAddGradsPSO;
    id<MTLComputePipelineState> _positionalAddWeightsGradsPSO;

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

        _positionalAddPSO = [self createPipelineStateWithFunctionName:@"positionalAdd"];
        _positionalAddGradsPSO = [self createPipelineStateWithFunctionName:@"positionalAddGrads"];
        _positionalAddWeightsGradsPSO = [self createPipelineStateWithFunctionName:@"positionalAddWeightsGrads"];
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
    uint depth = inputData.length / (sizeof(float) * colsCount * rowsCount);

    id<MTLComputeCommandEncoder> addPos = [commandBuffer computeCommandEncoder];
    [addPos setComputePipelineState:_positionalAddPSO];
    [addPos setBuffer:inputData offset:0 atIndex:0];
    [addPos setBuffer:weightsData offset:0 atIndex:1];
    [addPos setBuffer:outputData offset:0 atIndex:2];
    [addPos setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [addPos setBytes:&rowsCount length:sizeof(uint) atIndex:4];
    [addPos dispatchThreads:MTLSizeMake(colsCount, rowsCount, depth) threadsPerThreadgroup:threadgroupSize2D(_positionalAddPSO)];
    [addPos endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    uint depth = inputGrad.length / (sizeof(float) * colsCount * rowsCount);

    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_positionalAddGradsPSO];
    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:0];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:1];
    [calcInputGrads setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [calcInputGrads setBytes:&rowsCount length:sizeof(uint) atIndex:3];
    [calcInputGrads dispatchThreads:MTLSizeMake(colsCount, rowsCount, depth) threadsPerThreadgroup:threadgroupSize2D(_positionalAddGradsPSO)];
    [calcInputGrads endEncoding];

    id<MTLComputeCommandEncoder> calcWeightsGrads = [commandBuffer computeCommandEncoder];
    [calcWeightsGrads setComputePipelineState:_positionalAddWeightsGradsPSO];
    [calcWeightsGrads setBuffer:weightsGrad offset:0 atIndex:0];
    [calcWeightsGrads setBuffer:outputGrad offset:0 atIndex:1];
    [calcWeightsGrads setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [calcWeightsGrads setBytes:&rowsCount length:sizeof(uint) atIndex:3];
    [calcWeightsGrads setBytes:&depth length:sizeof(uint) atIndex:4];
    [calcWeightsGrads dispatchThreads:MTLSizeMake(colsCount, rowsCount, 1) threadsPerThreadgroup:threadgroupSize2D(_positionalAddWeightsGradsPSO)];
    [calcWeightsGrads endEncoding];
}

@end
