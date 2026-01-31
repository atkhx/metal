package gelunew

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* gelunewKernelCreate(void *device, const char *kernelSource) {
    return [[GeLuNewKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void gelunewForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData
) {
    [(__bridge GeLuNewKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData];
}

void gelunewBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputData,
    void *outputGrad
) {
    [(__bridge GeLuNewKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputData:(id<MTLBuffer>)outputData
        outputGrad:(id<MTLBuffer>)outputGrad];
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

func New(device *mtl.Device, input, output *num.Data) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	return &Kernel{
		kernelID: C.gelunewKernelCreate(device.GetID(), cKernelString),
		input:    input,
		output:   output,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	input    *num.Data
	output   *num.Data
}

func (op *Kernel) Forward(b *mtl.CommandBuffer) {
	C.gelunewForward(
		op.kernelID,
		b.GetID(),
		op.input.Data.GetID(),
		op.output.Data.GetID(),
	)
}

func (op *Kernel) Backward(b *mtl.CommandBuffer) {
	C.gelunewBackward(
		op.kernelID,
		b.GetID(),
		op.input.Data.GetID(),
		op.input.Grad.GetID(),
		op.output.Data.GetID(),
		op.output.Grad.GetID(),
	)
}
