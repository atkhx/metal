#ifndef SiluKernel_h
#define SiluKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol SiluKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad;

@end


@interface SiluKernelImpl : NSObject <SiluKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* SiluKernel_h */
