#ifndef RmsNormRowsOptKernel_h
#define RmsNormRowsOptKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol RmsNormRowsOptKernel <NSObject>
- (void)forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        rmsData:(id<MTLBuffer>)rmsData
        chunkSize:(uint)chunkSize;

- (void)backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        rmsData:(id<MTLBuffer>)rmsData
        rmsGrad:(id<MTLBuffer>)rmsGrad
        chunkSize:(uint)chunkSize;
@end

@interface RmsNormRowsOptKernelImpl : NSObject <RmsNormRowsOptKernel>
@property (nonatomic, strong) id<MTLLibrary> library;
- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
@end

#endif /* RmsNormRowsOptKernel_h */
