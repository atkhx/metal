#ifndef TrilMaskKernel_h
#define TrilMaskKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol TrilMaskKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        mask:(float)mask
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount;

@end


@interface TrilMaskKernelImpl : NSObject <TrilMaskKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* TrilMaskKernel_h */
