package rmsnormrowsopt

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* rmsNormRowsOptKernelCreate(void *device, const char *kernelSource) {
    return [[RmsNormRowsOptKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void rmsNormRowsOptForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *rmsData,
    uint chunkSize
) {
    [(__bridge RmsNormRowsOptKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        rmsData:(id<MTLBuffer>)rmsData
        chunkSize:chunkSize];
}

void rmsNormRowsOptBackward(
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
    [(__bridge RmsNormRowsOptKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
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
	if chunkSize <= 0 {
		panic("rmsnormrows_opt: chunkSize must be > 0")
	}
	if input.Dims.Length()%chunkSize != 0 {
		panic("rmsnormrows_opt: input length not divisible by chunkSize")
	}
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	rows := input.Dims.Length() / chunkSize
	if rows == 0 {
		panic("rmsnormrows_opt: rows count is 0")
	}
	return &Kernel{
		kernelID: C.rmsNormRowsOptKernelCreate(device.GetID(), cKernelString),
		input:    input,
		output:   output,
		chunk:    chunkSize,
		rms:      device.NewBufferEmptyFloatsBuffer(rows, mtl.ResourceStorageModeShared),
		rmsGrad:  device.NewBufferEmptyFloatsBuffer(rows, mtl.ResourceStorageModeShared),
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	input    *num.Data
	output   *num.Data
	chunk    int
	rms      *mtl.Buffer
	rmsGrad  *mtl.Buffer
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	if k.kernelID == nil {
		panic("rmsnormrows_opt: kernel is nil")
	}
	C.rmsNormRowsOptForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.rms.GetID(),
		C.uint(k.chunk),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	if k.kernelID == nil {
		panic("rmsnormrows_opt: kernel is nil")
	}
	C.rmsNormRowsOptBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.output.Data.GetID(),
		k.output.Grad.GetID(),
		k.rms.GetID(),
		k.rmsGrad.GetID(),
		C.uint(k.chunk),
	)
}
