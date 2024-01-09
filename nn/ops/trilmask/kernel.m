#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation TrilMaskKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _trilMaskPSO;
    id<MTLComputePipelineState> _trilMaskBwdPSO;

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

        _trilMaskPSO = [self createPipelineStateWithFunctionName:@"trilMask"];
        _trilMaskBwdPSO = [self createPipelineStateWithFunctionName:@"trilMaskBwd"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        mask:(float)mask
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    uint depth = inputData.length/(sizeof(float)*colsCount*rowsCount);
    id<MTLComputeCommandEncoder> trilMask = [commandBuffer computeCommandEncoder];
    [trilMask setComputePipelineState:_trilMaskPSO];
    [trilMask setBuffer:inputData offset:0 atIndex:0];
    [trilMask setBuffer:outputData offset:0 atIndex:1];
    [trilMask setBytes:&mask length:sizeof(float) atIndex:2];
    [trilMask setBytes:&colsCount length:sizeof(uint) atIndex:3];
    [trilMask setBytes:&rowsCount length:sizeof(uint) atIndex:4];
    [trilMask dispatchThreads:MTLSizeMake(colsCount, rowsCount, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [trilMask endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
{
    uint depth = inputGrad.length/(sizeof(float)*colsCount*rowsCount);
    id<MTLComputeCommandEncoder> trilMaskGrads = [commandBuffer computeCommandEncoder];

    [trilMaskGrads setComputePipelineState:_trilMaskBwdPSO];
    [trilMaskGrads setBuffer:inputGrad offset:0 atIndex:0];
    [trilMaskGrads setBuffer:outputGrad offset:0 atIndex:1];
    [trilMaskGrads setBytes:&colsCount length:sizeof(uint) atIndex:2];
    [trilMaskGrads setBytes:&rowsCount length:sizeof(uint) atIndex:3];
    [trilMaskGrads dispatchThreads:MTLSizeMake(colsCount, rowsCount, depth) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [trilMaskGrads endEncoding];
}

@end