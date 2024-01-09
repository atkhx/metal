package mulrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* mulRowsKernelCreate(void *device, const char *kernelSource) {
    return [[MulRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void mulRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
	uint chunkSize
) {

    [(__bridge MulRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        chunkSize:(uint)chunkSize
	];
}

void mulRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
	void *weightsData,
	void *weightsGrad,
    void *outputData,
    void *outputGrad,
	uint chunkSize
) {
    [(__bridge MulRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
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
	rowWidth int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.mulRowsKernelCreate(device.GetID(), cKernelString),

		device:   device,
		input:    input,
		weights:  weights,
		output:   output,
		rowWidth: rowWidth,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device   *mtl.Device
	input    *num.Data
	weights  *num.Data
	output   *num.Data
	rowWidth int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.mulRowsForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.weights.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.rowWidth),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.mulRowsBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.weights.Data.GetID(),
		k.weights.Grad.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.rowWidth),
	)
}
