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

@implementation MaxPoolKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _maxPoolPSO;
    id<MTLComputePipelineState> _maxPoolGradsPSO;

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

        _maxPoolPSO = [self createPipelineStateWithFunctionName:@"maxPoolForward"];
        _maxPoolGradsPSO = [self createPipelineStateWithFunctionName:@"maxPoolBackward"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        maskData:(id<MTLBuffer>)maskData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
        poolSize:(uint)poolSize
        stride:(uint)stride
        padding:(uint)padding
{
    uint depth = inputData.length / (sizeof(float) * inW * inH);

    id<MTLComputeCommandEncoder> maxPool = [commandBuffer computeCommandEncoder];
    [maxPool setComputePipelineState:_maxPoolPSO];
    [maxPool setBuffer:inputData offset:0 atIndex:0];
    [maxPool setBuffer:outputData offset:0 atIndex:1];
    [maxPool setBuffer:maskData offset:0 atIndex:2];
    [maxPool setBytes:&inW length:sizeof(uint) atIndex:3];
    [maxPool setBytes:&inH length:sizeof(uint) atIndex:4];
    [maxPool setBytes:&outW length:sizeof(uint) atIndex:5];
    [maxPool setBytes:&outH length:sizeof(uint) atIndex:6];
    [maxPool setBytes:&poolSize length:sizeof(uint) atIndex:7];
    [maxPool setBytes:&stride length:sizeof(uint) atIndex:8];
    [maxPool setBytes:&padding length:sizeof(uint) atIndex:9];
    [maxPool dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_maxPoolPSO)];
    [maxPool endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        maskData:(id<MTLBuffer>)maskData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
        poolSize:(uint)poolSize
        stride:(uint)stride
        padding:(uint)padding
{
    uint depth = inputGrad.length / (sizeof(float) * inW * inH);

    id<MTLComputeCommandEncoder> maxPoolGrads = [commandBuffer computeCommandEncoder];
    [maxPoolGrads setComputePipelineState:_maxPoolGradsPSO];
    [maxPoolGrads setBuffer:inputGrad offset:0 atIndex:0];
    [maxPoolGrads setBuffer:outputGrad offset:0 atIndex:1];
    [maxPoolGrads setBuffer:maskData offset:0 atIndex:2];
    [maxPoolGrads setBytes:&inW length:sizeof(uint) atIndex:3];
    [maxPoolGrads setBytes:&inH length:sizeof(uint) atIndex:4];
    [maxPoolGrads setBytes:&outW length:sizeof(uint) atIndex:5];
    [maxPoolGrads setBytes:&outH length:sizeof(uint) atIndex:6];
    [maxPoolGrads setBytes:&poolSize length:sizeof(uint) atIndex:7];
    [maxPoolGrads setBytes:&stride length:sizeof(uint) atIndex:8];
    [maxPoolGrads setBytes:&padding length:sizeof(uint) atIndex:9];
    [maxPoolGrads dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_maxPoolGradsPSO)];
    [maxPoolGrads endEncoding];
}

@end
