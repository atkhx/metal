package transpose

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* transposeKernelCreate(void *device, const char *kernelSource) {
    return [[TransposeKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void transposeForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	uint width,
	uint height
) {
    [(__bridge TransposeKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		width:(uint)width
        height:(uint)height
	];
}

void transposeBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
	uint width,
	uint height

) {
    [(__bridge TransposeKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
		width:(uint)width
        height:(uint)height
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
	width int,
	height int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.transposeKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,

		width:  width,
		height: height,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data

	width  int
	height int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.transposeForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.width),
		C.uint(k.height),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.transposeBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.width),
		C.uint(k.height),
	)
}
