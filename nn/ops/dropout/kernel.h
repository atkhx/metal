#ifndef DropoutKernel_h
#define DropoutKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol DropoutKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        randomData:(id<MTLBuffer>)randomData
        probability:(float)probability;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        randomData:(id<MTLBuffer>)randomData
        probability:(float)probability;

@end


@interface DropoutKernelImpl : NSObject <DropoutKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* DropoutKernel_h */
