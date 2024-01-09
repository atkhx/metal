package relu

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* reluKernelCreate(void *device, const char *kernelSource) {
    return [[ReluKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void reluForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData
) {

    [(__bridge ReluKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
	];
}

void reluBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputGrad
) {
    [(__bridge ReluKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
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
		kernelID: C.reluKernelCreate(device.GetID(), cKernelString),

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
	C.reluForward(k.kernelID, b.GetID(), k.input.Data.GetID(), k.output.Data.GetID())
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.reluBackward(k.kernelID, b.GetID(), k.input.Data.GetID(), k.input.Grad.GetID(), k.output.Grad.GetID())
}
