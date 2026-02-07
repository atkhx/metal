#ifndef MPSConvKernel_h
#define MPSConvKernel_h

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@protocol MPSConvKernel <NSObject>
- (id<MTLTexture>)inputTexture;
- (id<MTLTexture>)outputTexture;
- (id<MTLTexture>)outputGradTexture;
- (id<MTLTexture>)inputGradTexture;

- (void)encodeForward:(id<MTLCommandBuffer>)commandBuffer;
- (void)encodeBackward:(id<MTLCommandBuffer>)commandBuffer;

- (id<MTLBuffer>)weightsGradBuffer;
- (id<MTLBuffer>)biasGradBuffer;

- (void)reloadWeights;
@end

@interface MPSConvKernelImpl : NSObject <MPSConvKernel>
- (instancetype)initWithDevice:(id<MTLDevice>)device
                     inputWidth:(NSUInteger)inW
                    inputHeight:(NSUInteger)inH
                   inputChannels:(NSUInteger)inC
                  outputChannels:(NSUInteger)outC
                    kernelWidth:(NSUInteger)kW
                   kernelHeight:(NSUInteger)kH
                          stride:(NSUInteger)stride
                         padding:(NSUInteger)padding
                       batchSize:(NSUInteger)batchSize
                          weights:(id<MTLBuffer>)weights
                            biases:(id<MTLBuffer>)biases;
@end

#endif /* MPSConvKernel_h */
