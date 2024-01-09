#ifndef RmsNormRowsKernel_h
#define RmsNormRowsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol RmsNormRowsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        rmsData:(id<MTLBuffer>)rmsData
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        rmsData:(id<MTLBuffer>)rmsData
        rmsGrad:(id<MTLBuffer>)rmsGrad
        chunkSize:(uint)chunkSize;

@end


@interface RmsNormRowsKernelImpl : NSObject <RmsNormRowsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* RmsNormRowsKernel_h */
