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

@implementation VAEKLKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _vaeKLPSO;
    id<MTLComputePipelineState> _vaeKLGradsPSO;

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

        _vaeKLPSO = [self createPipelineStateWithFunctionName:@"vaeKLForward"];
        _vaeKLGradsPSO = [self createPipelineStateWithFunctionName:@"vaeKLBackward"];
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
{
    uint depth = outputData.length / (sizeof(float) * outW * outH);

    id<MTLComputeCommandEncoder> vaeKL = [commandBuffer computeCommandEncoder];
    [vaeKL setComputePipelineState:_vaeKLPSO];
    [vaeKL setBuffer:inputData offset:0 atIndex:0];
    [vaeKL setBuffer:outputData offset:0 atIndex:1];
    [vaeKL setBytes:&inW length:sizeof(uint) atIndex:2];
    [vaeKL setBytes:&inH length:sizeof(uint) atIndex:3];
    [vaeKL setBytes:&outW length:sizeof(uint) atIndex:4];
    [vaeKL setBytes:&outH length:sizeof(uint) atIndex:5];
    [vaeKL dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_vaeKLPSO)];
    [vaeKL endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
{
    uint depth = outputGrad.length / (sizeof(float) * outW * outH);

    id<MTLComputeCommandEncoder> vaeKLGrads = [commandBuffer computeCommandEncoder];
    [vaeKLGrads setComputePipelineState:_vaeKLGradsPSO];
    [vaeKLGrads setBuffer:inputData offset:0 atIndex:0];
    [vaeKLGrads setBuffer:inputGrad offset:0 atIndex:1];
    [vaeKLGrads setBuffer:outputGrad offset:0 atIndex:2];
    [vaeKLGrads setBytes:&inW length:sizeof(uint) atIndex:3];
    [vaeKLGrads setBytes:&inH length:sizeof(uint) atIndex:4];
    [vaeKLGrads setBytes:&outW length:sizeof(uint) atIndex:5];
    [vaeKLGrads setBytes:&outH length:sizeof(uint) atIndex:6];
    [vaeKLGrads dispatchThreads:MTLSizeMake(outW, outH, depth) threadsPerThreadgroup:threadgroupSize2D(_vaeKLGradsPSO)];
    [vaeKLGrads endEncoding];
}

@end
