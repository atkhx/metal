#ifndef MaxPoolKernel_h
#define MaxPoolKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MaxPoolKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

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
        padding:(uint)padding;

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
        padding:(uint)padding;

@end

@interface MaxPoolKernelImpl : NSObject <MaxPoolKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MaxPoolKernel_h */
