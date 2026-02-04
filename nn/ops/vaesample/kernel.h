#ifndef VAESampleKernel_h
#define VAESampleKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol VAESampleKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        epsData:(id<MTLBuffer>)epsData
        randomData:(id<MTLBuffer>)randomData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        epsData:(id<MTLBuffer>)epsData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH;

@end


@interface VAESampleKernelImpl : NSObject <VAESampleKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* VAESampleKernel_h */
