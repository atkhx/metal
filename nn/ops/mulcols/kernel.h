#ifndef MulColsKernel_h
#define MulColsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MulColsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        rowWidth:(uint)rowWidth
        colHeight:(uint)colHeight;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        rowWidth:(uint)rowWidth
        colHeight:(uint)colHeight;

@end


@interface MulColsKernelImpl : NSObject <MulColsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MulColsKernel_h */
