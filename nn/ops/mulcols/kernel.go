package mulcols

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* mulColsKernelCreate(void *device, const char *kernelSource) {
    return [[MulColsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void mulColsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
	uint rowWidth,
	uint colHeight
) {

    [(__bridge MulColsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
		rowWidth:(uint)rowWidth
        colHeight:(uint)colHeight
	];
}

void mulColsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
	void *weightsData,
	void *weightsGrad,
    void *outputData,
    void *outputGrad,
	uint rowWidth,
	uint colHeight
) {
    [(__bridge MulColsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
		rowWidth:(uint)rowWidth
		colHeight:(uint)colHeight
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
	colHeight int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.mulColsKernelCreate(device.GetID(), cKernelString),

		device:    device,
		input:     input,
		weights:   weights,
		output:    output,
		rowWidth:  rowWidth,
		colHeight: colHeight,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device    *mtl.Device
	input     *num.Data
	weights   *num.Data
	output    *num.Data
	rowWidth  int
	colHeight int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.mulColsForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.weights.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.rowWidth),
		C.uint(k.colHeight),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.mulColsBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.weights.Data.GetID(),
		k.weights.Grad.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.rowWidth),
		C.uint(k.colHeight),
	)
}
