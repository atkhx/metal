#ifndef NegLogLikelihoodKernel_h
#define NegLogLikelihoodKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol NegLogLikelihoodKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        softmax:(id<MTLBuffer>)softmax
        output:(id<MTLBuffer>)output
        targets:(id<MTLBuffer>)targets
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        targets:(id<MTLBuffer>)targets
        softmax:(id<MTLBuffer>)softmax
        nllGrad:(id<MTLBuffer>)nllGrad
        chunkSize:(uint)chunkSize;

@end

@interface NegLogLikelihoodKernelImpl : NSObject <NegLogLikelihoodKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* NegLogLikelihoodKernel_h */
