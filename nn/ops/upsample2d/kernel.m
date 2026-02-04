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

@implementation UpSample2DKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _upSamplePSO;
    id<MTLComputePipelineState> _upSampleGradsPSO;

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

        _upSamplePSO = [self createPipelineStateWithFunctionName:@"upSample2DForward"];
        _upSampleGradsPSO = [self createPipelineStateWithFunctionName:@"upSample2DBackward"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
        scale:(uint)scale
{
    uint depth = inputData.length / (sizeof(float) * inW * inH);

    id<MTLComputeCommandEncoder> upSample = [commandBuffer computeCommandEncoder];
    [upSample setComputePipelineState:_upSamplePSO];
    [upSample setBuffer:inputData offset:0 atIndex:0];
    [upSample setBuffer:outputData offset:0 atIndex:1];
    [upSample setBytes:&inW length:sizeof(uint) atIndex:2];
    [upSample setBytes:&inH length:sizeof(uint) atIndex:3];
    [upSample setBytes:&outW length:sizeof(uint) atIndex:4];
    [upSample setBytes:&outH length:sizeof(uint) atIndex:5];
    [upSample setBytes:&scale length:sizeof(uint) atIndex:6];
    [upSample dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_upSamplePSO)];
    [upSample endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
        scale:(uint)scale
{
    uint depth = inputGrad.length / (sizeof(float) * inW * inH);

    id<MTLComputeCommandEncoder> upSampleGrads = [commandBuffer computeCommandEncoder];
    [upSampleGrads setComputePipelineState:_upSampleGradsPSO];
    [upSampleGrads setBuffer:inputGrad offset:0 atIndex:0];
    [upSampleGrads setBuffer:outputGrad offset:0 atIndex:1];
    [upSampleGrads setBytes:&inW length:sizeof(uint) atIndex:2];
    [upSampleGrads setBytes:&inH length:sizeof(uint) atIndex:3];
    [upSampleGrads setBytes:&outW length:sizeof(uint) atIndex:4];
    [upSampleGrads setBytes:&outH length:sizeof(uint) atIndex:5];
    [upSampleGrads setBytes:&scale length:sizeof(uint) atIndex:6];
    [upSampleGrads dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_upSampleGradsPSO)];
    [upSampleGrads endEncoding];
}

@end
