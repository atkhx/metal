#ifndef EmbeddingsKernel_h
#define EmbeddingsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol EmbeddingsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount;

@end


@interface EmbeddingsKernelImpl : NSObject <EmbeddingsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* EmbeddingsKernel_h */
