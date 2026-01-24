package clamp

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* clampKernelCreate(void *device, const char *kernelSource) {
    return [[ClampKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void clampForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
	float minValue,
	float maxValue
) {
    [(__bridge ClampKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
		minValue:(float)minValue
		maxValue:(float)maxValue
	];
}

void clampBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputGrad,
	float minValue,
	float maxValue
) {
    [(__bridge ClampKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
		minValue:(float)minValue
		maxValue:(float)maxValue
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
	minValue float32,
	maxValue float32,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.clampKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,

		minValue: minValue,
		maxValue: maxValue,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data

	minValue float32
	maxValue float32
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.clampForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		C.float(k.minValue),
		C.float(k.maxValue),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.clampBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		C.float(k.minValue),
		C.float(k.maxValue),
	)
}
