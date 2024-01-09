package mean

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* meanKernelCreate(void *device, const char *kernelSource) {
    return [[MeanKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void meanForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	uint chunkSize
) {
    [(__bridge MeanKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
	];
}

void meanBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
	uint chunkSize
) {
    [(__bridge MeanKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        chunkSize:(uint)chunkSize
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
	chunkSize int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.meanKernelCreate(device.GetID(), cKernelString),

		device:    device,
		input:     input,
		output:    output,
		chunkSize: chunkSize,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device    *mtl.Device
	input     *num.Data
	output    *num.Data
	chunkSize int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.meanForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.chunkSize),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.meanBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.chunkSize),
	)
}
