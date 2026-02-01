#ifndef PositionalAddKernel_h
#define PositionalAddKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol PositionalAddKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

@end

@interface PositionalAddKernelImpl : NSObject <PositionalAddKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* PositionalAddKernel_h */
