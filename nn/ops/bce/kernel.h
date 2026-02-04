#ifndef BCEKernel_h
#define BCEKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol BCEKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        targetData:(id<MTLBuffer>)targetData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        targetData:(id<MTLBuffer>)targetData
        outputGrad:(id<MTLBuffer>)outputGrad;

@end


@interface BCEKernelImpl : NSObject <BCEKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* BCEKernel_h */
