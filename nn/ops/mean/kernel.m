#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation MeanKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _meanByRowsPSO;
    id<MTLComputePipelineState> _meanByRowsGradsPSO;

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

        _meanByRowsPSO      = [self createPipelineStateWithFunctionName:@"meanByRows"];
        _meanByRowsGradsPSO = [self createPipelineStateWithFunctionName:@"meanByRowsGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
{
    uint rowsCount = inputData.length/(sizeof(float)*chunkSize);

    id<MTLComputeCommandEncoder> meanByRows = [commandBuffer computeCommandEncoder];
    [meanByRows setComputePipelineState:_meanByRowsPSO];
    [meanByRows setBuffer:inputData offset:0 atIndex:0];
    [meanByRows setBuffer:outputData offset:0 atIndex:1];
    [meanByRows setBytes:&chunkSize length:sizeof(uint) atIndex:2];
    [meanByRows dispatchThreads:MTLSizeMake(1, rowsCount, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [meanByRows endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize
{
    id<MTLComputeCommandEncoder> meanGrads = [commandBuffer computeCommandEncoder];
    [meanGrads setComputePipelineState:_meanByRowsGradsPSO];
    [meanGrads setBuffer:inputGrad offset:0 atIndex:0];
    [meanGrads setBuffer:outputGrad offset:0 atIndex:1];
    [meanGrads setBytes:&chunkSize length:sizeof(uint) atIndex:2];
    [meanGrads dispatchThreads:MTLSizeMake(inputGrad.length/sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [meanGrads endEncoding];
}

@end