package layernormrowsopt

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* layerNormRowsOptKernelCreate(void *device, const char *kernelSource) {
    return [[LayerNormRowsOptKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void layerNormRowsOptForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *meanData,
    void *invStdData,
    uint width,
    float eps,
    uint rowsCount
) {
    [(__bridge LayerNormRowsOptKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        meanData:(id<MTLBuffer>)meanData
        invStdData:(id<MTLBuffer>)invStdData
        width:width
        eps:eps
        rowsCount:rowsCount];
}

void layerNormRowsOptBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputGrad,
    void *meanData,
    void *invStdData,
    void *sumDy,
    void *sumDyXmu,
    uint width,
    uint rowsCount
) {
    [(__bridge LayerNormRowsOptKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        meanData:(id<MTLBuffer>)meanData
        invStdData:(id<MTLBuffer>)invStdData
        sumDy:(id<MTLBuffer>)sumDy
        sumDyXmu:(id<MTLBuffer>)sumDyXmu
        width:width
        rowsCount:rowsCount];
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
	eps float32,
) *Kernel {
	if width <= 0 {
		panic("layernormrows_opt: width must be > 0")
	}
	if input.Dims.Length()%width != 0 {
		panic("layernormrows_opt: input length not divisible by width")
	}
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	rows := input.Dims.Length() / width
	if rows == 0 {
		panic("layernormrows_opt: rows count is 0")
	}
	return &Kernel{
		kernelID: C.layerNormRowsOptKernelCreate(device.GetID(), cKernelString),
		input:    input,
		output:   output,
		width:    width,
		rows:     rows,
		eps:      eps,
		mean:     device.NewBufferEmptyFloatsBuffer(rows, mtl.ResourceStorageModeShared),
		invStd:   device.NewBufferEmptyFloatsBuffer(rows, mtl.ResourceStorageModeShared),
		sumDy:    device.NewBufferEmptyFloatsBuffer(rows, mtl.ResourceStorageModeShared),
		sumDyXmu: device.NewBufferEmptyFloatsBuffer(rows, mtl.ResourceStorageModeShared),
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	input    *num.Data
	output   *num.Data
	width    int
	rows     int
	eps      float32
	mean     *mtl.Buffer
	invStd   *mtl.Buffer
	sumDy    *mtl.Buffer
	sumDyXmu *mtl.Buffer
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	if k.kernelID == nil {
		panic("layernormrows_opt: kernel is nil")
	}
	C.layerNormRowsOptForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.mean.GetID(),
		k.invStd.GetID(),
		C.uint(k.width),
		C.float(k.eps),
		C.uint(k.rows),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	if k.kernelID == nil {
		panic("layernormrows_opt: kernel is nil")
	}
	C.layerNormRowsOptBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		k.mean.GetID(),
		k.invStd.GetID(),
		k.sumDy.GetID(),
		k.sumDyXmu.GetID(),
		C.uint(k.width),
		C.uint(k.rows),
	)
}
