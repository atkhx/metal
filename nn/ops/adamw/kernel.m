#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

struct Parameters {
    uint width;
};

@implementation MPSAdamWImpl {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _updateWithAdamPSO;
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
        _updateWithAdamPSO = [self createPipelineStateWithFunctionName:@"updateWithAdam"];
    }
    return self;
}

- (void) updateWithAdam:(id<MTLCommandBuffer>)commandBuffer
        dataBuffer:(id<MTLBuffer>)dataBuffer
        gradBuffer:(id<MTLBuffer>)gradBuffer
        mBuffer:(id<MTLBuffer>)mBuffer
        vBuffer:(id<MTLBuffer>)vBuffer
        beta1:(float)beta1
        beta2:(float)beta2
        beta1powIterationLR:(float)beta1powIterationLR
        beta2powIteration:(float)beta2powIteration
{
    id<MTLComputeCommandEncoder> updateWithAdam = [commandBuffer computeCommandEncoder];
    [updateWithAdam setComputePipelineState:_updateWithAdamPSO];

    [updateWithAdam setBuffer:dataBuffer offset:0 atIndex:0];
    [updateWithAdam setBuffer:gradBuffer offset:0 atIndex:1];

    [updateWithAdam setBuffer:mBuffer offset:0 atIndex:2];
    [updateWithAdam setBuffer:vBuffer offset:0 atIndex:3];

    [updateWithAdam setBytes:&beta1 length:sizeof(float) atIndex:4];
    [updateWithAdam setBytes:&beta2 length:sizeof(float) atIndex:5];

    [updateWithAdam setBytes:&beta1powIterationLR length:sizeof(float) atIndex:6];
    [updateWithAdam setBytes:&beta2powIteration length:sizeof(float) atIndex:7];

    [updateWithAdam dispatchThreads:MTLSizeMake(dataBuffer.length / sizeof(float), 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [updateWithAdam endEncoding];
}

@end