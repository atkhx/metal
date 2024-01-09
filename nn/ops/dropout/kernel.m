#import "kernel.h"
#import <Foundation/Foundation.h>
#include <stdio.h>

@implementation DropoutKernelImpl {
    id<MTLDevice> _device;

    id<MTLComputePipelineState> _dropoutPSO;
    id<MTLComputePipelineState> _dropoutGradsPSO;

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

        _dropoutPSO      = [self createPipelineStateWithFunctionName:@"dropout"];
        _dropoutGradsPSO = [self createPipelineStateWithFunctionName:@"dropoutGrads"];
    }
    return self;
}

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        randomData:(id<MTLBuffer>)randomData
        probability:(float)probability
{
    id<MTLComputeCommandEncoder> dropout = [commandBuffer computeCommandEncoder];

    [dropout setComputePipelineState:_dropoutPSO];
    [dropout setBuffer:inputData offset:0 atIndex:0];
    [dropout setBuffer:outputData offset:0 atIndex:1];
    [dropout setBuffer:randomData offset:0 atIndex:2];
    [dropout setBytes:&probability length:sizeof(float) atIndex:3];

    [dropout dispatchThreads:MTLSizeMake(outputData.length / sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [dropout endEncoding];
}

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        randomData:(id<MTLBuffer>)randomData
        probability:(float)probability
{
    id<MTLComputeCommandEncoder> dropoutGrads = [commandBuffer computeCommandEncoder];

    [dropoutGrads setComputePipelineState:_dropoutGradsPSO];
    [dropoutGrads setBuffer:inputGrad offset:0 atIndex:0];
    [dropoutGrads setBuffer:outputGrad offset:0 atIndex:1];
    [dropoutGrads setBuffer:randomData offset:0 atIndex:2];
    [dropoutGrads setBytes:&probability length:sizeof(float) atIndex:3];

    [dropoutGrads dispatchThreads:MTLSizeMake(inputGrad.length / sizeof(float), 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [dropoutGrads endEncoding];
}

@end