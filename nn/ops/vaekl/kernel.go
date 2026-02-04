package vaekl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* vaeKLKernelCreate(void *device, const char *kernelSource) {
    return [[VAEKLKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void vaeKLForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    uint inW,
    uint inH,
    uint outW,
    uint outH
) {
    [(__bridge VAEKLKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
		inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inW:inW
        inH:inH
        outW:outW
        outH:outH
	];
}

void vaeKLBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputGrad,
    uint inW,
    uint inH,
    uint outW,
    uint outH
) {
    [(__bridge VAEKLKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
		inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inW:inW
        inH:inH
        outW:outW
        outH:outH
	];
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
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.vaeKLKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.vaeKLForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.vaeKLBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
	)
}
