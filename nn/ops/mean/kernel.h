#ifndef MeanKernel_h
#define MeanKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MeanKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize;

@end


@interface MeanKernelImpl : NSObject <MeanKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MeanKernel_h */
