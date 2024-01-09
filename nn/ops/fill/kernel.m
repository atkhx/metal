#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

struct Parameters {
    uint width;
};

@implementation fillKernelImpl {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _fillPSO;
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
        _fillPSO = [self createPipelineStateWithFunctionName:@"fill"];
    }
    return self;
}

- (void) fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length
{
    id<MTLComputeCommandEncoder> fill = [commandBuffer computeCommandEncoder];

    [fill setComputePipelineState:_fillPSO];
    [fill setBuffer:dstBuffer offset:offset atIndex:0];
    [fill setBytes:&value length:sizeof(float) atIndex:1];
    [fill dispatchThreads:MTLSizeMake(length / sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [fill endEncoding];
}

@end