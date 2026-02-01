package positionaladd

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* positionalAddKernelCreate(void *device, const char *kernelSource) {
    return [[PositionalAddKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void positionalAddForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
    uint colsCount,
    uint rowsCount
) {
    [(__bridge PositionalAddKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        colsCount:colsCount
        rowsCount:rowsCount];
}

void positionalAddBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *weightsGrad,
    void *outputGrad,
    uint colsCount,
    uint rowsCount
) {
    [(__bridge PositionalAddKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsGrad:(id<MTLBuffer>)weightsGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:colsCount
        rowsCount:rowsCount];
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
	colsCount int,
	rowsCount int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	return &Kernel{
		kernelID: C.positionalAddKernelCreate(device.GetID(), cKernelString),
		input:    input,
		weights:  weights,
		output:   output,
		cols:     colsCount,
		rows:     rowsCount,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	input    *num.Data
	weights  *num.Data
	output   *num.Data
	cols     int
	rows     int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.positionalAddForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.weights.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.cols),
		C.uint(k.rows),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.positionalAddBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.weights.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.cols),
		C.uint(k.rows),
	)
}
