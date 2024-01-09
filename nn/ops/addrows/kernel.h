#ifndef AddRowsKernel_h
#define AddRowsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol AddRowsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize;

@end


@interface AddRowsKernelImpl : NSObject <AddRowsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* AddRowsKernel_h */
