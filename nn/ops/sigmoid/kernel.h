#ifndef SigmoidKernel_h
#define SigmoidKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol SigmoidKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad;

@end


@interface SigmoidKernelImpl : NSObject <SigmoidKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* SigmoidKernel_h */
