#ifndef MPSPackKernel_h
#define MPSPackKernel_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MPSPackKernel <NSObject>
- (void)pack:(id<MTLCommandBuffer>)commandBuffer
       input:(id<MTLBuffer>)input
      output:(id<MTLTexture>)output
        width:(uint)width
       height:(uint)height
     channels:(uint)channels
    batchSize:(uint)batchSize
       slices:(uint)slices;

- (void)unpack:(id<MTLCommandBuffer>)commandBuffer
        input:(id<MTLTexture>)input
       output:(id<MTLBuffer>)output
         width:(uint)width
        height:(uint)height
      channels:(uint)channels
     batchSize:(uint)batchSize
        slices:(uint)slices;
@end

@interface MPSPackKernelImpl : NSObject <MPSPackKernel>
- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;
@end

#endif /* MPSPackKernel_h */
