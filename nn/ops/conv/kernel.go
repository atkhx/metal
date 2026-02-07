package conv

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* mpsConvKernelCreate(
    void *device,
    NSUInteger inW,
    NSUInteger inH,
    NSUInteger inC,
    NSUInteger outC,
    NSUInteger kW,
    NSUInteger kH,
    NSUInteger stride,
    NSUInteger padding,
    NSUInteger batchSize,
    void *weights,
    void *biases
) {
    return [[MPSConvKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
                                          inputWidth:inW
                                         inputHeight:inH
                                        inputChannels:inC
                                       outputChannels:outC
                                         kernelWidth:kW
                                        kernelHeight:kH
                                               stride:stride
                                              padding:padding
                                            batchSize:batchSize
                                               weights:(id<MTLBuffer>)weights
                                                 biases:(id<MTLBuffer>)biases];
}

void* mpsConvKernelGetInputTexture(void *kernel) {
    return [(__bridge MPSConvKernelImpl*)kernel inputTexture];
}
void* mpsConvKernelGetOutputTexture(void *kernel) {
    return [(__bridge MPSConvKernelImpl*)kernel outputTexture];
}
void* mpsConvKernelGetOutputGradTexture(void *kernel) {
    return [(__bridge MPSConvKernelImpl*)kernel outputGradTexture];
}
void* mpsConvKernelGetInputGradTexture(void *kernel) {
    return [(__bridge MPSConvKernelImpl*)kernel inputGradTexture];
}

void mpsConvKernelEncodeForward(void *kernel, void *commandBuffer) {
    [(__bridge MPSConvKernelImpl*)kernel encodeForward:(id<MTLCommandBuffer>)commandBuffer];
}

void mpsConvKernelEncodeBackward(void *kernel, void *commandBuffer) {
    [(__bridge MPSConvKernelImpl*)kernel encodeBackward:(id<MTLCommandBuffer>)commandBuffer];
}

void* mpsConvKernelGetWeightsGradBuffer(void *kernel) {
    return [(__bridge MPSConvKernelImpl*)kernel weightsGradBuffer];
}
void* mpsConvKernelGetBiasGradBuffer(void *kernel) {
    return [(__bridge MPSConvKernelImpl*)kernel biasGradBuffer];
}

void mpsConvKernelReloadWeights(void *kernel) {
    [(__bridge MPSConvKernelImpl*)kernel reloadWeights];
}

*/
import "C"
import (
	"unsafe"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/ops/mpspack"
)

type Kernel struct {
	kernelID unsafe.Pointer

	input   *num.Data
	weights *num.Data
	biases  *num.Data
	output  *num.Data

	inW  int
	inH  int
	inC  int
	outC int

	packer *mpspack.Kernel
}

func New(
	device *mtl.Device,
	input *num.Data,
	weights *num.Data,
	biases *num.Data,
	output *num.Data,
	filtersCount int,
	batchSize int,
	padding int,
	stride int,
) *Kernel {
	inW := input.Dims.W
	inH := input.Dims.H
	inC := input.Dims.D / batchSize
	outC := filtersCount
	kW := weights.Dims.W
	kH := weights.Dims.H

	var biasesID unsafe.Pointer
	if biases != nil {
		biasesID = biases.Data.GetID()
	}

	return &Kernel{
		kernelID: C.mpsConvKernelCreate(
			device.GetID(),
			C.NSUInteger(inW),
			C.NSUInteger(inH),
			C.NSUInteger(inC),
			C.NSUInteger(outC),
			C.NSUInteger(kW),
			C.NSUInteger(kH),
			C.NSUInteger(stride),
			C.NSUInteger(padding),
			C.NSUInteger(batchSize),
			weights.Data.GetID(),
			biasesID,
		),
		input:   input,
		weights: weights,
		biases:  biases,
		output:  output,
		inW:     inW,
		inH:     inH,
		inC:     inC,
		outC:    outC,
		packer:  mpspack.New(device),
	}
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.mpsConvKernelReloadWeights(k.kernelID)
	inTex := C.mpsConvKernelGetInputTexture(k.kernelID)
	outTex := C.mpsConvKernelGetOutputTexture(k.kernelID)

	k.packer.Pack(b, k.input.Data, inTex, k.inW, k.inH, k.inC, k.input.Dims.D/k.inC)
	C.mpsConvKernelEncodeForward(k.kernelID, b.GetID())
	k.packer.Unpack(b, outTex, k.output.Data, k.output.Dims.W, k.output.Dims.H, k.outC, k.output.Dims.D/k.outC)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.mpsConvKernelReloadWeights(k.kernelID)
	inTex := C.mpsConvKernelGetInputTexture(k.kernelID)
	outGradTex := C.mpsConvKernelGetOutputGradTexture(k.kernelID)
	inGradTex := C.mpsConvKernelGetInputGradTexture(k.kernelID)

	k.packer.Pack(b, k.input.Data, inTex, k.inW, k.inH, k.inC, k.input.Dims.D/k.inC)
	k.packer.Pack(b, k.output.Grad, outGradTex, k.output.Dims.W, k.output.Dims.H, k.outC, k.output.Dims.D/k.outC)

	C.mpsConvKernelEncodeBackward(k.kernelID, b.GetID())

	k.packer.Unpack(b, inGradTex, k.input.Grad, k.inW, k.inH, k.inC, k.input.Dims.D/k.inC)

	wGrad := mtl.CreateBuffer(unsafe.Pointer(C.mpsConvKernelGetWeightsGradBuffer(k.kernelID)))
	bGrad := mtl.CreateBuffer(unsafe.Pointer(C.mpsConvKernelGetBiasGradBuffer(k.kernelID)))
	enc := b.GetMTLBlitCommandEncoder()
	enc.CopyBuffer(wGrad, 0, k.weights.Grad, 0, k.weights.Grad.GetLengthBytes())
	enc.CopyBuffer(bGrad, 0, k.biases.Grad, 0, k.biases.Grad.GetLengthBytes())
	enc.EndEncoding()
}

func (k *Kernel) ReloadWeights() {
	C.mpsConvKernelReloadWeights(k.kernelID)
}
