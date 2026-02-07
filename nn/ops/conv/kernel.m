#import "kernel.h"

@interface MPSConvDataSource : NSObject <MPSCNNConvolutionDataSource>
@property (nonatomic, assign) void *weightsPtr;
@property (nonatomic, assign) float *biasPtr;
@property (nonatomic, strong) MPSCNNConvolutionDescriptor *desc;
@property (nonatomic, copy) NSString *labelName;
@end

@implementation MPSConvDataSource
- (MPSDataType)dataType { return MPSDataTypeFloat32; }
- (MPSCNNConvolutionDescriptor *)descriptor { return _desc; }
- (void *)weights { return _weightsPtr; }
- (float *)biasTerms { return _biasPtr; }
- (BOOL)load { return YES; }
- (void)purge { }
- (NSString *)label { return _labelName; }
- (id)copyWithZone:(NSZone *)zone { return self; }
@end

@implementation MPSConvKernelImpl {
    id<MTLDevice> _device;

    NSUInteger _inW, _inH, _inC, _outC, _kW, _kH, _stride, _padding, _batch;

    MPSImageDescriptor *_inDesc;
    MPSImageDescriptor *_outDesc;

    MPSImage *_inImage;
    MPSImage *_outImage;
    MPSImage *_srcGradImage;
    MPSImage *_dstGradImage;

    MPSCNNConvolution *_conv;
    MPSCNNConvolutionGradient *_convGrad;
    MPSCNNConvolutionGradientState *_gradState;
    MPSConvDataSource *_dataSource;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device
                     inputWidth:(NSUInteger)inW
                    inputHeight:(NSUInteger)inH
                   inputChannels:(NSUInteger)inC
                  outputChannels:(NSUInteger)outC
                    kernelWidth:(NSUInteger)kW
                   kernelHeight:(NSUInteger)kH
                          stride:(NSUInteger)stride
                         padding:(NSUInteger)padding
                       batchSize:(NSUInteger)batchSize
                          weights:(id<MTLBuffer>)weights
                            biases:(id<MTLBuffer>)biases
{
    self = [super init];
    if (self) {
        _device = device;

        _inW = inW; _inH = inH; _inC = inC; _outC = outC;
        _kW = kW; _kH = kH; _stride = stride; _padding = padding; _batch = batchSize;

        MPSCNNConvolutionDescriptor *desc =
            [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kW
                                                                      kernelHeight:kH
                                                              inputFeatureChannels:inC
                                                             outputFeatureChannels:outC
                                                                      neuronFilter:nil];
        desc.strideInPixelsX = stride;
        desc.strideInPixelsY = stride;

        MPSNNPaddingMethod padMethod = (_padding > 0) ? MPSNNPaddingMethodSizeSame : MPSNNPaddingMethodSizeValidOnly;
        id<MPSNNPadding> pad = [MPSNNDefaultPadding paddingWithMethod:padMethod];

        _dataSource = [MPSConvDataSource new];
        _dataSource.weightsPtr = [weights contents];
        _dataSource.biasPtr = (biases != nil) ? (float *)[biases contents] : NULL;
        _dataSource.desc = desc;
        _dataSource.labelName = @"mpsconv";

        _conv = [[MPSCNNConvolution alloc] initWithDevice:_device weights:_dataSource];
        _conv.padding = pad;

        _convGrad = [[MPSCNNConvolutionGradient alloc] initWithDevice:_device weights:_dataSource];
        _convGrad.gradientOption = MPSCNNConvolutionGradientOptionAll;

        _inDesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                                 width:inW
                                                                height:inH
                                                       featureChannels:inC
                                                        numberOfImages:batchSize
                                                                 usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

        NSUInteger outW = (inW - kW + 2 * padding) / stride + 1;
        NSUInteger outH = (inH - kH + 2 * padding) / stride + 1;
        _outDesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                                  width:outW
                                                                 height:outH
                                                        featureChannels:outC
                                                         numberOfImages:batchSize
                                                                  usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

        _inImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:_inDesc];
        _outImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:_outDesc];
        _srcGradImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:_outDesc];
        _dstGradImage = [[MPSImage alloc] initWithDevice:_device imageDescriptor:_inDesc];
    }
    return self;
}

- (void)reloadWeights {
    [_conv reloadWeightsAndBiasesFromDataSource];
    [_convGrad reloadWeightsAndBiasesFromDataSource];
}

- (id<MTLTexture>)inputTexture { return [_inImage texture]; }
- (id<MTLTexture>)outputTexture { return [_outImage texture]; }
- (id<MTLTexture>)outputGradTexture { return [_srcGradImage texture]; }
- (id<MTLTexture>)inputGradTexture { return [_dstGradImage texture]; }

- (void)encodeForward:(id<MTLCommandBuffer>)commandBuffer {
    _gradState = [_conv resultStateForSourceImage:_inImage sourceStates:nil destinationImage:_outImage];
    [_conv encodeToCommandBuffer:commandBuffer sourceImage:_inImage destinationImage:_outImage];
}

- (void)encodeBackward:(id<MTLCommandBuffer>)commandBuffer {
    [_convGrad encodeToCommandBuffer:commandBuffer
                      sourceGradient:_srcGradImage
                         sourceImage:_inImage
                       gradientState:_gradState
                 destinationGradient:_dstGradImage];
}

- (id<MTLBuffer>)weightsGradBuffer { return [_gradState gradientForWeights]; }
- (id<MTLBuffer>)biasGradBuffer { return [_gradState gradientForBiases]; }

@end
