#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation RmsNormRowsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _calcRMSByRowsPSO;
    id<MTLComputePipelineState> _normByRMSPSO;

    id<MTLComputePipelineState> _calcRMSGradsPSO;
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

        _calcRMSByRowsPSO = [self createPipelineStateWithFunctionName:@"calcRMSByRows"];
        _normByRMSPSO     = [self createPipelineStateWithFunctionName:@"normByRMS"];

        _calcRMSGradsPSO   = [self createPipelineStateWithFunctionName:@"calcRMSGrads"];
        _calcInputGradsPSO = [self createPipelineStateWithFunctionName:@"calcInputGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        rmsData:(id<MTLBuffer>)rmsData
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> calcRMSByRows = [commandBuffer computeCommandEncoder];
    [calcRMSByRows setComputePipelineState:_calcRMSByRowsPSO];
    [calcRMSByRows setBuffer:inputData offset:0 atIndex:0];
    [calcRMSByRows setBuffer:rmsData offset:0 atIndex:1];
    [calcRMSByRows setBytes:&chunkSize length:sizeof(uint) atIndex:2];
    [calcRMSByRows dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcRMSByRows endEncoding];

    id<MTLComputeCommandEncoder> normByRMS = [commandBuffer computeCommandEncoder];
    [normByRMS setComputePipelineState:_normByRMSPSO];
    [normByRMS setBuffer:inputData offset:0 atIndex:0];
    [normByRMS setBuffer:outputData offset:0 atIndex:1];
    [normByRMS setBuffer:rmsData offset:0 atIndex:2];
    [normByRMS setBytes:&chunkSize length:sizeof(uint) atIndex:3];
    [normByRMS dispatchThreads:MTLSizeMake(chunkSize, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [normByRMS endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        rmsData:(id<MTLBuffer>)rmsData
        rmsGrad:(id<MTLBuffer>)rmsGrad
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length / (sizeof(float) * chunkSize);

    id<MTLComputeCommandEncoder> calcRMSGrads = [commandBuffer computeCommandEncoder];
    [calcRMSGrads setComputePipelineState:_calcRMSGradsPSO];
    [calcRMSGrads setBuffer:rmsData offset:0 atIndex:0];
    [calcRMSGrads setBuffer:rmsGrad offset:0 atIndex:1];
    [calcRMSGrads setBuffer:outputData offset:0 atIndex:2];
    [calcRMSGrads setBuffer:outputGrad offset:0 atIndex:3];
    [calcRMSGrads setBytes:&chunkSize length:sizeof(uint) atIndex:4];
    [calcRMSGrads dispatchThreads:MTLSizeMake(rowsCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcRMSGrads endEncoding];

    id<MTLComputeCommandEncoder> calcInputGrads = [commandBuffer computeCommandEncoder];
    [calcInputGrads setComputePipelineState:_calcInputGradsPSO];
    [calcInputGrads setBuffer:inputData offset:0 atIndex:0];
    [calcInputGrads setBuffer:inputGrad offset:0 atIndex:1];
    [calcInputGrads setBuffer:outputGrad offset:0 atIndex:2];
    [calcInputGrads setBuffer:rmsData offset:0 atIndex:3];
    [calcInputGrads setBuffer:rmsGrad offset:0 atIndex:4];
    [calcInputGrads setBytes:&chunkSize length:sizeof(uint) atIndex:5];
    [calcInputGrads dispatchThreads:MTLSizeMake(inputGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [calcInputGrads endEncoding];
}

@end