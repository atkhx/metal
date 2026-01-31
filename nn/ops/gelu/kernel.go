package gelu

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* geluKernelCreate(void *device, const char *kernelSource) {
    return [[GeluKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void geluForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData
) {
    [(__bridge GeluKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData];
}

void geluBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputData,
    void *outputGrad
) {
    [(__bridge GeluKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad];
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
		kernelID: C.geluKernelCreate(device.GetID(), cKernelString),
		input:    input,
		output:   output,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	input    *num.Data
	output   *num.Data
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.geluForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.geluBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
	)
}
