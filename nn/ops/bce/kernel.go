package bce

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* bceKernelCreate(void *device, const char *kernelSource) {
    return [[BCEKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void bceForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *targetData,
    void *outputData
) {

    [(__bridge BCEKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        targetData:(id<MTLBuffer>)targetData
        outputData:(id<MTLBuffer>)outputData
	];
}

void bceBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *targetData,
    void *outputGrad
) {
    [(__bridge BCEKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        targetData:(id<MTLBuffer>)targetData
        outputGrad:(id<MTLBuffer>)outputGrad
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
	targets *num.Data,
	output *num.Data,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.bceKernelCreate(device.GetID(), cKernelString),

		device:  device,
		input:   input,
		targets: targets,
		output:  output,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device  *mtl.Device
	input   *num.Data
	targets *num.Data
	output  *num.Data
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.bceForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.targets.Data.GetID(),
		k.output.Data.GetID(),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.bceBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.targets.Data.GetID(),
		k.output.Grad.GetID(),
	)
}
