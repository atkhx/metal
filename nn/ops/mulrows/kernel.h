#ifndef MulRowsKernel_h
#define MulRowsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MulRowsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize;

@end


@interface MulRowsKernelImpl : NSObject <MulRowsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MulRowsKernel_h */
