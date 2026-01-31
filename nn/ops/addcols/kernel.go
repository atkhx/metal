package addcols

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* addColsKernelCreate(void *device, const char *kernelSource) {
    return [[AddColsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void addColsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *outputData,
    uint colsCount,
    uint rowsCount
) {
    [(__bridge AddColsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        outputData:(id<MTLBuffer>)outputData
        colsCount:colsCount
        rowsCount:rowsCount];
}

void addColsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *weightsGrad,
    void *outputGrad,
    uint colsCount,
    uint rowsCount
) {
    [(__bridge AddColsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
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
		kernelID: C.addColsKernelCreate(device.GetID(), cKernelString),
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
	C.addColsForward(
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
	C.addColsBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.weights.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.cols),
		C.uint(k.rows),
	)
}
