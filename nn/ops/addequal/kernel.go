package addequal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* addEqualKernelCreate(void *device, const char *kernelSource) {
    return [[addEqualKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void addEqualForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData
) {
    [(__bridge addEqualKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
	];
}

void addEqualBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
	void *weightsGrad,
    void *outputGrad
) {
    [(__bridge addEqualKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
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
	weights *num.Data,
	output *num.Data,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	return &Kernel{
		kernelID: C.addEqualKernelCreate(device.GetID(), cKernelString),

		device:  device,
		input:   input,
		weights: weights,
		output:  output,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	device   *mtl.Device
	input    *num.Data
	weights  *num.Data
	output   *num.Data
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.addEqualForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.weights.Data.GetID(),
		k.output.Data.GetID(),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.addEqualBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.weights.Grad.GetID(),
		k.output.Grad.GetID(),
	)
}
