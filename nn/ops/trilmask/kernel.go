package trilmask

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* trilMaskKernelCreate(void *device, const char *kernelSource) {
    return [[TrilMaskKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void trilMaskForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	float mask,
	uint colsCount,
	uint rowsCount
) {

    [(__bridge TrilMaskKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		mask:(float)mask
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
	];
}

void trilMaskBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
	uint colsCount,
	uint rowsCount
) {

    [(__bridge TrilMaskKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        colsCount:(uint)colsCount
        rowsCount:(uint)rowsCount
	];
}

*/
import "C"
import (
	_ "embed"
	"math"
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
	colsCount int,
	rowsCount int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	return &Kernel{
		kernelID: C.trilMaskKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,

		colsCount: colsCount,
		rowsCount: rowsCount,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data

	colsCount int
	rowsCount int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.trilMaskForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		C.float(float32(math.Inf(-1))),
		C.uint(k.colsCount),
		C.uint(k.rowsCount),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.trilMaskBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.colsCount),
		C.uint(k.rowsCount),
	)
}
