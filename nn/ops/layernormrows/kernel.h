#ifndef LayerNormRowsKernel_h
#define LayerNormRowsKernel_h

#import <Metal/Metal.h>

@protocol LayerNormRowsKernel <NSObject>

- (void) forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        meanData:(id<MTLBuffer>)meanData
        invStdData:(id<MTLBuffer>)invStdData
        width:(uint)width
        eps:(float)eps
        rowsCount:(uint)rowsCount;

- (void) backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        meanData:(id<MTLBuffer>)meanData
        invStdData:(id<MTLBuffer>)invStdData
        sumDy:(id<MTLBuffer>)sumDy
        sumDyXmu:(id<MTLBuffer>)sumDyXmu
        width:(uint)width
        rowsCount:(uint)rowsCount;

@end

@interface LayerNormRowsKernelImpl : NSObject <LayerNormRowsKernel>

@property (nonatomic, strong) id<MTLLibrary> library;

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

@end

#endif /* LayerNormRowsKernel_h */
