package maxpool

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* maxPoolKernelCreate(void *device, const char *kernelSource) {
    return [[MaxPoolKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void maxPoolForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *maskData,
    uint inW,
    uint inH,
    uint outW,
    uint outH,
    uint poolSize,
    uint stride,
    uint padding
) {
    [(__bridge MaxPoolKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        maskData:(id<MTLBuffer>)maskData
        inW:inW
        inH:inH
        outW:outW
        outH:outH
        poolSize:poolSize
        stride:stride
        padding:padding];
}

void maxPoolBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    void *maskData,
    uint inW,
    uint inH,
    uint outW,
    uint outH,
    uint poolSize,
    uint stride,
    uint padding
) {
    [(__bridge MaxPoolKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        maskData:(id<MTLBuffer>)maskData
        inW:inW
        inH:inH
        outW:outW
        outH:outH
        poolSize:poolSize
        stride:stride
        padding:padding];
}
*/
import "C"
import (
	_ "embed"
	"unsafe"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

//go:embed kernel.metal
var metalFunctions string

func New(
	device *mtl.Device,
	input *num.Data,
	output *num.Data,
	poolSize int,
	stride int,
	padding int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	mask := device.NewBufferEmptyFloatsBuffer(output.Dims.Length(), mtl.ResourceStorageModeShared)

	return &Kernel{
		kernelID: C.maxPoolKernelCreate(device.GetID(), cKernelString),
		input:    input,
		output:   output,
		mask:     mask,
		poolSize: poolSize,
		stride:   stride,
		padding:  padding,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	input    *num.Data
	output   *num.Data
	mask     *mtl.Buffer
	poolSize int
	stride   int
	padding  int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.maxPoolForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.mask.GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
		C.uint(k.poolSize),
		C.uint(k.stride),
		C.uint(k.padding),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.maxPoolBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		k.mask.GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
		C.uint(k.poolSize),
		C.uint(k.stride),
		C.uint(k.padding),
	)
}
