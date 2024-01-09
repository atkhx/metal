#ifndef fillKernel_h
#define fillKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol fillKernel <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length;

@end


@interface fillKernelImpl : NSObject <fillKernel>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* fillKernel_h */
