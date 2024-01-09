#ifndef ropeColsKernel_h
#define ropeColsKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol ropeColsKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength;

@end


@interface ropeColsKernelImpl : NSObject <ropeColsKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* ropeColsKernel_h */
