package mulequal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* mulEqualKernelCreate(void *device, const char *kernelSource) {
    return [[MulEqualKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void mulEqualForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData
) {

    [(__bridge MulEqualKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
	];
}

void mulEqualBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
	void *weightsData,
	void *weightsGrad,
    void *outputData,
    void *outputGrad
) {
    [(__bridge MulEqualKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
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
		kernelID: C.mulEqualKernelCreate(device.GetID(), cKernelString),

		device:  device,
		input:   input,
		weights: weights,
		output:  output,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device  *mtl.Device
	input   *num.Data
	weights *num.Data
	output  *num.Data
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.mulEqualForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.weights.Data.GetID(),
		k.output.Data.GetID(),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.mulEqualBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.weights.Data.GetID(),
		k.weights.Grad.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
	)
}
