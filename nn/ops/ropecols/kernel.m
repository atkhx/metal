#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation ropeColsKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _ropeColsPSO;
    id<MTLComputePipelineState> _ropeColsGradsPSO;

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
    
        _ropeColsPSO      = [self createPipelineStateWithFunctionName:@"ropeCols"];
        _ropeColsGradsPSO = [self createPipelineStateWithFunctionName:@"ropeColsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
{
    uint batchSize = inputData.length/(sizeof(float) * featuresCount * contextLength);
    id<MTLComputeCommandEncoder> ropeCols = [commandBuffer computeCommandEncoder];

    [ropeCols setComputePipelineState:_ropeColsPSO];
    [ropeCols setBuffer:inputData offset:0 atIndex:0];
    [ropeCols setBuffer:outputData offset:0 atIndex:1];
    [ropeCols setBytes:&featuresCount length:sizeof(uint) atIndex:2];
    [ropeCols setBytes:&headSize length:sizeof(uint) atIndex:3];
    [ropeCols setBytes:&contextLength length:sizeof(uint) atIndex:4];
    [ropeCols dispatchThreads:MTLSizeMake(contextLength, featuresCount/2, batchSize) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ropeCols endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
{
    uint batchSize = inputGrad.length/(sizeof(float) * featuresCount * contextLength);
    id<MTLComputeCommandEncoder> ropeColsGrads = [commandBuffer computeCommandEncoder];

    [ropeColsGrads setComputePipelineState:_ropeColsGradsPSO];
    [ropeColsGrads setBuffer:inputGrad offset:0 atIndex:0];
    [ropeColsGrads setBuffer:outputGrad offset:0 atIndex:1];
    [ropeColsGrads setBytes:&featuresCount length:sizeof(uint) atIndex:2];
    [ropeColsGrads setBytes:&headSize length:sizeof(uint) atIndex:3];
    [ropeColsGrads setBytes:&contextLength length:sizeof(uint) atIndex:4];
    [ropeColsGrads dispatchThreads:MTLSizeMake(contextLength, featuresCount/2, batchSize) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [ropeColsGrads endEncoding];
}

@end