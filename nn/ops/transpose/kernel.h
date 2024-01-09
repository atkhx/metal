#ifndef TransposeKernel_h
#define TransposeKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol TransposeKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        width:(uint)width
        height:(uint)height;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        width:(uint)width
        height:(uint)height;

@end


@interface TransposeKernelImpl : NSObject <TransposeKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* TransposeKernel_h */
