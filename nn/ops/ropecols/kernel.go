package ropecols

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* ropeColsKernelCreate(void *device, const char *kernelSource) {
    return [[ropeColsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void ropeColsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    uint featuresCount,
    uint headSize,
    uint contextLength
) {

    [(__bridge ropeColsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
	];
}

void ropeColsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    uint featuresCount,
    uint headSize,
    uint contextLength
) {
    [(__bridge ropeColsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
		featuresCount:(uint)featuresCount
        headSize:(uint)headSize
        contextLength:(uint)contextLength
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
	featuresCount int,
	headSize int,
	contextLength int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.ropeColsKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,

		featuresCount: featuresCount,
		headSize:      headSize,
		contextLength: contextLength,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data

	featuresCount int
	headSize      int
	contextLength int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.ropeColsForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.featuresCount),
		C.uint(k.headSize),
		C.uint(k.contextLength),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.ropeColsBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.featuresCount),
		C.uint(k.headSize),
		C.uint(k.contextLength),
	)
}
