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

@implementation VAESampleKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _vaeSamplePSO;
    id<MTLComputePipelineState> _vaeSampleGradsPSO;

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

        _vaeSamplePSO = [self createPipelineStateWithFunctionName:@"vaeSampleForward"];
        _vaeSampleGradsPSO = [self createPipelineStateWithFunctionName:@"vaeSampleBackward"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        epsData:(id<MTLBuffer>)epsData
        randomData:(id<MTLBuffer>)randomData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
{
    uint depth = outputData.length / (sizeof(float) * outW * outH);

    id<MTLComputeCommandEncoder> vaeSample = [commandBuffer computeCommandEncoder];
    [vaeSample setComputePipelineState:_vaeSamplePSO];
    [vaeSample setBuffer:inputData offset:0 atIndex:0];
    [vaeSample setBuffer:outputData offset:0 atIndex:1];
    [vaeSample setBuffer:epsData offset:0 atIndex:2];
    [vaeSample setBuffer:randomData offset:0 atIndex:3];
    [vaeSample setBytes:&inW length:sizeof(uint) atIndex:4];
    [vaeSample setBytes:&inH length:sizeof(uint) atIndex:5];
    [vaeSample setBytes:&outW length:sizeof(uint) atIndex:6];
    [vaeSample setBytes:&outH length:sizeof(uint) atIndex:7];
    [vaeSample dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_vaeSamplePSO)];
    [vaeSample endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        epsData:(id<MTLBuffer>)epsData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
{
    uint depth = outputGrad.length / (sizeof(float) * outW * outH);

    id<MTLComputeCommandEncoder> vaeSampleGrads = [commandBuffer computeCommandEncoder];
    [vaeSampleGrads setComputePipelineState:_vaeSampleGradsPSO];
    [vaeSampleGrads setBuffer:inputData offset:0 atIndex:0];
    [vaeSampleGrads setBuffer:inputGrad offset:0 atIndex:1];
    [vaeSampleGrads setBuffer:outputGrad offset:0 atIndex:2];
    [vaeSampleGrads setBuffer:epsData offset:0 atIndex:3];
    [vaeSampleGrads setBytes:&inW length:sizeof(uint) atIndex:4];
    [vaeSampleGrads setBytes:&inH length:sizeof(uint) atIndex:5];
    [vaeSampleGrads setBytes:&outW length:sizeof(uint) atIndex:6];
    [vaeSampleGrads setBytes:&outH length:sizeof(uint) atIndex:7];
    [vaeSampleGrads dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_vaeSampleGradsPSO)];
    [vaeSampleGrads endEncoding];
}

@end
