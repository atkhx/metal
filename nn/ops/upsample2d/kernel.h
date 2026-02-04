#ifndef UpSample2DKernel_h
#define UpSample2DKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol UpSample2DKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
        scale:(uint)scale;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inW:(uint)inW
        inH:(uint)inH
        outW:(uint)outW
        outH:(uint)outH
        scale:(uint)scale;

@end


@interface UpSample2DKernelImpl : NSObject <UpSample2DKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* UpSample2DKernel_h */
