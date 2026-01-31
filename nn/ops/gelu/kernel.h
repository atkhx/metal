#ifndef GeluKernel_h
#define GeluKernel_h

#import <Metal/Metal.h>

@protocol GeluKernel <NSObject>

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad;

@end

@interface GeluKernelImpl : NSObject <GeluKernel>

@property (nonatomic, strong) id<MTLLibrary> library;

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

@end

#endif /* GeluKernel_h */
