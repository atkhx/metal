#ifndef addEqualKernel_h
#define addEqualKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol addEqualKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad;

@end


@interface addEqualKernelImpl : NSObject <addEqualKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* addEqualKernel_h */
