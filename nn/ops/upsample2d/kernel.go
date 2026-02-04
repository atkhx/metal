package upsample2d

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* upSample2DKernelCreate(void *device, const char *kernelSource) {
    return [[UpSample2DKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void upSample2DForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    uint inW,
    uint inH,
    uint outW,
    uint outH,
    uint scale
) {
    [(__bridge UpSample2DKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
		inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        inW:inW
        inH:inH
        outW:outW
        outH:outH
        scale:scale
	];
}

void upSample2DBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    uint inW,
    uint inH,
    uint outW,
    uint outH,
    uint scale
) {
    [(__bridge UpSample2DKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
		inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        inW:inW
        inH:inH
        outW:outW
        outH:outH
        scale:scale
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
	scale int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.upSample2DKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,
		scale:  scale,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data

	scale int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.upSample2DForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
		C.uint(k.scale),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.upSample2DBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
		C.uint(k.scale),
	)
}
