package nllpos

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* nllKernelCreate(void *device, const char *kernelSource) {
    return [[NegLogLikelihoodKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void nllPosForward(
    void *kernel,
    void *commandBuffer,
    void *softmax,
    void *output,
    void *targets,
    uint chunkSize
) {
    [(__bridge NegLogLikelihoodKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        softmax:(id<MTLBuffer>)softmax
        output:(id<MTLBuffer>)output
        targets:(id<MTLBuffer>)targets
        chunkSize:(uint)chunkSize
	];
}

void nllPosBackward(
    void *kernel,
    void *commandBuffer,
    void *outputData,
    void *outputGrad,
    void *targets,
    void *softmax,
    void *nllGrad,
    uint chunkSize
) {
    [(__bridge NegLogLikelihoodKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        targets:(id<MTLBuffer>)targets
        softmax:(id<MTLBuffer>)softmax
        nllGrad:(id<MTLBuffer>)nllGrad
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
	output *num.Data,
	targets *num.Data,
	chunkSize int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.nllKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,

		targets:   targets,
		chunkSize: chunkSize,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device  *mtl.Device
	input   *num.Data
	output  *num.Data
	targets *num.Data

	chunkSize int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.nllPosForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.targets.Data.GetID(),
		C.uint(k.chunkSize),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.nllPosBackward(
		k.kernelID,
		b.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
		k.targets.Data.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		C.uint(k.chunkSize),
	)
}
