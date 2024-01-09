#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation TransposeKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _transposePSO;
    id<MTLComputePipelineState> _transposeGradsPSO;

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

        _transposePSO       = [self createPipelineStateWithFunctionName:@"transpose"];
        _transposeGradsPSO = [self createPipelineStateWithFunctionName:@"transposeGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height
{
    uint square = width * height;
    uint depth = outputData.length/(sizeof(float)*square);

    id<MTLComputeCommandEncoder> transpose = [commandBuffer computeCommandEncoder];
    [transpose setComputePipelineState:_transposePSO];
    [transpose setBuffer:inputData offset:0 atIndex:0];
    [transpose setBuffer:outputData offset:0 atIndex:1];
    [transpose setBytes:&width length:sizeof(uint) atIndex:2];
    [transpose setBytes:&height length:sizeof(uint) atIndex:3];
    [transpose setBytes:&square length:sizeof(uint) atIndex:4];
    [transpose dispatchThreads:MTLSizeMake(width, height, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [transpose endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        width:(uint)width
        height:(uint)height
{
    uint square = width * height;
    uint depth = inputGrad.length/(sizeof(float)*square);

    id<MTLComputeCommandEncoder> transposeGrads = [commandBuffer computeCommandEncoder];
    [transposeGrads setComputePipelineState:_transposeGradsPSO];
    [transposeGrads setBuffer:inputGrad offset:0 atIndex:0];
    [transposeGrads setBuffer:outputGrad offset:0 atIndex:1];
    [transposeGrads setBytes:&width length:sizeof(uint) atIndex:2];
    [transposeGrads setBytes:&height length:sizeof(uint) atIndex:3];
    [transposeGrads setBytes:&square length:sizeof(uint) atIndex:4];
    [transposeGrads dispatchThreads:MTLSizeMake(width, height, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [transposeGrads endEncoding];
}

@end