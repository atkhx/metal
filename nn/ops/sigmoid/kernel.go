package sigmoid

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* sigmoidKernelCreate(void *device, const char *kernelSource) {
    return [[SigmoidKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void sigmoidForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData
) {

    [(__bridge SigmoidKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
	];
}

void sigmoidBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputData,
    void *outputGrad
) {
    [(__bridge SigmoidKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
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
	output *num.Data,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.sigmoidKernelCreate(device.GetID(), cKernelString),

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
	C.sigmoidForward(k.kernelID, b.GetID(), k.input.Data.GetID(), k.output.Data.GetID())
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.sigmoidBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
	)
}
