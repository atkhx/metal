#ifndef AddColsKernel_h
#define AddColsKernel_h

#import <Metal/Metal.h>

@protocol AddColsKernel <NSObject>

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

@interface AddColsKernelImpl : NSObject <AddColsKernel>

@property (nonatomic, strong) id<MTLLibrary> library;

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

@end

#endif /* AddColsKernel_h */
