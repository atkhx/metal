#ifndef MulEqualKernel_h
#define MulEqualKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MulEqualKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad;

@end


@interface MulEqualKernelImpl : NSObject <MulEqualKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MulEqualKernel_h */
