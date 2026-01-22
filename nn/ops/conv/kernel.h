#ifndef convKernel_h
#define convKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

@protocol convKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        biasesData:(id<MTLBuffer>)biasesData
        outputData:(id<MTLBuffer>)outputData
        iDims:(MTLSize)iDims
        wDims:(MTLSize)wDims
        oDims:(MTLSize)oDims
        filtersCount:(uint)filtersCount
        batchSize:(uint)batchSize
        padding:(uint)padding
        stride:(uint)stride;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        biasesGrad:(id<MTLBuffer>)biasesGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        iDims:(MTLSize)iDims
        wDims:(MTLSize)wDims
        oDims:(MTLSize)oDims
        filtersCount:(uint)filtersCount
        batchSize:(uint)batchSize
        padding:(uint)padding
        stride:(uint)stride;

@end


@interface convKernelImpl : NSObject <convKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* convKernel_h */
