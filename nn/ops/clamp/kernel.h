#ifndef ClampKernel_h
#define ClampKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol ClampKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        minValue:(float)minValue
        maxValue:(float)maxValue;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        minValue:(float)minValue
        maxValue:(float)maxValue;

@end

@interface ClampKernelImpl : NSObject <ClampKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* ClampKernel_h */
