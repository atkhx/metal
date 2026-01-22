package conv

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* convKernelCreate(void *device, const char *kernelSource) {
    return [[convKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void convForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *weightsData,
    void *biasesData,
    void *outputData,
	MTLSize iDims,
	MTLSize wDims,
	MTLSize oDims,
	uint filtersCount,
	uint batchSize,
	uint padding,
    uint stride
) {
    [(__bridge convKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        weightsData:(id<MTLBuffer>)weightsData
        biasesData:(id<MTLBuffer>)biasesData
        outputData:(id<MTLBuffer>)outputData
		iDims:(MTLSize)iDims
		wDims:(MTLSize)wDims
		oDims:(MTLSize)oDims
		filtersCount:(uint)filtersCount
		batchSize:(uint)batchSize
		padding:(uint)padding
		stride:(uint)stride
	];
}

void convBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
	void *weightsData,
	void *weightsGrad,
	void *biasesGrad,
    void *outputGrad,
	MTLSize iDims,
	MTLSize wDims,
	MTLSize oDims,
	uint filtersCount,
	uint batchSize,
	uint padding,
    uint stride
) {
    [(__bridge convKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        weightsData:(id<MTLBuffer>)weightsData
        weightsGrad:(id<MTLBuffer>)weightsGrad
        biasesGrad:(id<MTLBuffer>)biasesGrad
        outputGrad:(id<MTLBuffer>)outputGrad
		iDims:(MTLSize)iDims
		wDims:(MTLSize)wDims
		oDims:(MTLSize)oDims
		filtersCount:(uint)filtersCount
		batchSize:(uint)batchSize
		padding:(uint)padding
		stride:(uint)stride
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
	weights *num.Data,
	biases *num.Data,
	output *num.Data,

	filtersCount int,
	batchSize int,
	padding int,
	stride int,

) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	return &Kernel{
		kernelID: C.convKernelCreate(device.GetID(), cKernelString),

		device:  device,
		input:   input,
		weights: weights,
		biases:  biases,
		output:  output,

		filtersCount: filtersCount,
		batchSize:    batchSize,
		padding:      padding,
		stride:       stride,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	device   *mtl.Device
	input    *num.Data
	weights  *num.Data
	biases   *num.Data
	output   *num.Data

	filtersCount int
	batchSize    int
	padding      int
	stride       int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.convForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.weights.Data.GetID(),
		k.biases.Data.GetID(),
		k.output.Data.GetID(),
		MTLSizeToC(k.input.Dims),
		MTLSizeToC(k.weights.Dims),
		MTLSizeToC(k.output.Dims),
		C.uint(k.filtersCount),
		C.uint(k.batchSize),
		C.uint(k.padding),
		C.uint(k.stride),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.convBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.weights.Data.GetID(),
		k.weights.Grad.GetID(),
		k.biases.Grad.GetID(),
		k.output.Grad.GetID(),
		MTLSizeToC(k.input.Dims),
		MTLSizeToC(k.weights.Dims),
		MTLSizeToC(k.output.Dims),
		C.uint(k.filtersCount),
		C.uint(k.batchSize),
		C.uint(k.padding),
		C.uint(k.stride),
	)
}

func MTLSizeToC(s mtl.MTLSize) C.MTLSize {
	return C.MTLSizeMake(C.ulong(s.W), C.ulong(s.H), C.ulong(s.D))
}
