#import "kernel.h"

static inline MTLSize threadgroupSize2D(id<MTLComputePipelineState> pso) {
    NSUInteger w = pso.threadExecutionWidth;
    NSUInteger h = pso.maxTotalThreadsPerThreadgroup / w;
    if (h > 1) {
        h = 1;
    }
    return MTLSizeMake(w, h, 1);
}

@interface MPSPackKernelImpl ()
@property (nonatomic, strong) id<MTLLibrary> library;
@end

@implementation MPSPackKernelImpl {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _packPSO;
    id<MTLComputePipelineState> _unpackPSO;
    NSError *error;
}

- (id<MTLComputePipelineState>)createPipelineStateWithFunctionName:(NSString *)functionName {
    id<MTLFunction> function = [self.library newFunctionWithName:functionName];
    if (!function) {
        printf("Failed to load function %s!\n", [functionName UTF8String]);
        return nil;
    }
    id<MTLComputePipelineState> pipelineState = [_device newComputePipelineStateWithFunction:function error:&error];
    if (error != nil) {
        const char *errorCString = [[error localizedDescription] UTF8String];
        printf("Failed to create pipeline state: %s\n", errorCString);
        return nil;
    }
    return pipelineState;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device kernelSource:(NSString*)kernelSource {
    self = [super init];
    if (self) {
        _device = device;
        self.library = [_device newLibraryWithSource:kernelSource options:nil error:&error];
        _packPSO = [self createPipelineStateWithFunctionName:@"packCHWToTexture"];
        _unpackPSO = [self createPipelineStateWithFunctionName:@"unpackTextureToCHW"];
    }
    return self;
}

- (void)pack:(id<MTLCommandBuffer>)commandBuffer
       input:(id<MTLBuffer>)input
      output:(id<MTLTexture>)output
        width:(uint)width
       height:(uint)height
     channels:(uint)channels
    batchSize:(uint)batchSize
       slices:(uint)slices
{
    id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
    [enc setComputePipelineState:_packPSO];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setTexture:output atIndex:0];
    [enc setBytes:&width length:sizeof(uint) atIndex:1];
    [enc setBytes:&height length:sizeof(uint) atIndex:2];
    [enc setBytes:&channels length:sizeof(uint) atIndex:3];
    [enc setBytes:&batchSize length:sizeof(uint) atIndex:4];
    [enc setBytes:&slices length:sizeof(uint) atIndex:5];

    [enc dispatchThreads:MTLSizeMake(width, height, batchSize*slices)
  threadsPerThreadgroup:threadgroupSize2D(_packPSO)];
    [enc endEncoding];
}

- (void)unpack:(id<MTLCommandBuffer>)commandBuffer
        input:(id<MTLTexture>)input
       output:(id<MTLBuffer>)output
         width:(uint)width
        height:(uint)height
      channels:(uint)channels
     batchSize:(uint)batchSize
        slices:(uint)slices
{
    id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
    [enc setComputePipelineState:_unpackPSO];
    [enc setTexture:input atIndex:0];
    [enc setBuffer:output offset:0 atIndex:0];
    [enc setBytes:&width length:sizeof(uint) atIndex:1];
    [enc setBytes:&height length:sizeof(uint) atIndex:2];
    [enc setBytes:&channels length:sizeof(uint) atIndex:3];
    [enc setBytes:&batchSize length:sizeof(uint) atIndex:4];
    [enc setBytes:&slices length:sizeof(uint) atIndex:5];

    [enc dispatchThreads:MTLSizeMake(width, height, batchSize*slices)
  threadsPerThreadgroup:threadgroupSize2D(_unpackPSO)];
    [enc endEncoding];
}

@end
