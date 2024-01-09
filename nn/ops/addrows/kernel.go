package addrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* addRowsKernelCreate(void *device, const char *kernelSource) {
    return [[AddRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void addRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
	uint chunkSize
) {
    [(__bridge AddRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
	];
}

void addRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
	void *weightsGrad,
    void *outputGrad,
	uint chunkSize
) {
    [(__bridge AddRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
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
	weights *num.Data,
	output *num.Data,
	chunkSize int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	return &Kernel{
		kernelID: C.addRowsKernelCreate(device.GetID(), cKernelString),

		device:    device,
		input:     input,
		weights:   weights,
		output:    output,
		chunkSize: chunkSize,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device  *mtl.Device
	input   *num.Data
	weights *num.Data
	output  *num.Data

	chunkSize int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.addRowsForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.weights.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.chunkSize),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.addRowsBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.weights.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.chunkSize),
	)
}
