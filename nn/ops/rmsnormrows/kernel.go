package rmsnormrows

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* RmsNormRowsKernelCreate(void *device, const char *kernelSource) {
    return [[RmsNormRowsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void rmsRowsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *rmsData,
    uint chunkSize
) {
    [(__bridge RmsNormRowsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        rmsData:(id<MTLBuffer>)rmsData
        chunkSize:chunkSize];
}

void rmsRowsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputData,
    void *outputGrad,
    void *rmsData,
    void *rmsGrad,
    uint chunkSize
) {
    [(__bridge RmsNormRowsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad
        rmsData:(id<MTLBuffer>)rmsData
        rmsGrad:(id<MTLBuffer>)rmsGrad
        chunkSize:chunkSize];
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
	chunkSize int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	return &Kernel{
		kernelID: C.RmsNormRowsKernelCreate(device.GetID(), cKernelString),

		device:    device,
		input:     input,
		output:    output,
		chunkSize: chunkSize,

		rmsData: device.NewBufferEmptyFloatsBuffer(input.Dims.Length()/chunkSize, mtl.ResourceStorageModeShared),
		rmsGrad: device.NewBufferEmptyFloatsBuffer(input.Dims.Length()/chunkSize, mtl.ResourceStorageModeShared),
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device    *mtl.Device
	input     *num.Data
	output    *num.Data
	chunkSize int

	rmsData *mtl.Buffer
	rmsGrad *mtl.Buffer
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.rmsRowsForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.rmsData.GetID(),
		C.uint(k.chunkSize),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.rmsRowsBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
		k.rmsData.GetID(),
		k.rmsGrad.GetID(),
		C.uint(k.chunkSize),
	)
}
