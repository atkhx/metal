#ifndef MPSAdamW_h
#define MPSAdamW_h

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@protocol MPSAdamW <NSObject>

- (instancetype) initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource;

- (void) updateWithAdam:(id<MTLCommandBuffer>)commandBuffer
        dataBuffer:(id<MTLBuffer>)dataBuffer
        gradBuffer:(id<MTLBuffer>)gradBuffer
        mBuffer:(id<MTLBuffer>)mBuffer
        vBuffer:(id<MTLBuffer>)vBuffer
        beta1:(float)beta1
        beta2:(float)beta2
        beta1powIterationLR:(float)beta1powIterationLR
        beta2powIteration:(float)beta2powIteration;

@end


@interface MPSAdamWImpl : NSObject <MPSAdamW>
    @property (nonatomic, strong) id<MTLLibrary> library;
@end

#endif /* MPSAdamW_h */
