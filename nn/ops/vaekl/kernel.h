#ifndef VAEKLKernel_h
#define VAEKLKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol VAEKLKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH;

@end


@interface VAEKLKernelImpl : NSObject <VAEKLKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* VAEKLKernel_h */
