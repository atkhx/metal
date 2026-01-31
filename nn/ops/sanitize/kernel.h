#ifndef SanitizeKernel_h
#define SanitizeKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol SanitizeKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad;

@end

@interface SanitizeKernelImpl : NSObject <SanitizeKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* SanitizeKernel_h */
