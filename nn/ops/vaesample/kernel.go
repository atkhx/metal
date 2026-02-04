package vaesample

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* vaeSampleKernelCreate(void *device, const char *kernelSource) {
    return [[VAESampleKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void vaeSampleForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *epsData,
    void *randomData,
    uint inW,
    uint inH,
    uint outW,
    uint outH
) {
    [(__bridge VAESampleKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
		inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        epsData:(id<MTLBuffer>)epsData
        randomData:(id<MTLBuffer>)randomData
        inW:inW
        inH:inH
        outW:outW
        outH:outH
	];
}

void vaeSampleBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *inputGrad,
    void *outputGrad,
    void *epsData,
    uint inW,
    uint inH,
    uint outW,
    uint outH
) {
    [(__bridge VAESampleKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
		inputData:(id<MTLBuffer>)inputData
        inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        epsData:(id<MTLBuffer>)epsData
        inW:inW
        inH:inH
        outW:outW
        outH:outH
	];
}

*/
import "C"
import (
	_ "embed"
	"time"
	"unsafe"

	"github.com/atkhx/metal/mps"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

//go:embed kernel.metal
var metalFunctions string

func New(
	device *mtl.Device,
	input *num.Data,
	output *num.Data,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	distribution := mps.CreateMatrixRandomDistributionDescriptor(0, 1)
	randomizer := mps.CreateMatrixRandomMTGP32(device, distribution, uint64(time.Now().UnixNano()))

	randomBuffer := device.NewBufferEmptyFloatsBuffer(output.Dims.Length()*2, mtl.ResourceStorageModeShared)
	randomDescriptor := mps.CreateMatrixDescriptorFloat32(
		output.Dims.W*2,
		output.Dims.H*output.Dims.D,
		1,
		output.Dims.W*output.Dims.H*output.Dims.D*2,
	)
	randomMatrix := mps.CreateMatrixWithBuffer(randomDescriptor, randomBuffer, 0)

	epsBuffer := device.NewBufferEmptyFloatsBuffer(output.Dims.Length(), mtl.ResourceStorageModeShared)

	return &Kernel{
		kernelID: C.vaeSampleKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,

		randomizer:   randomizer,
		randomMatrix: randomMatrix,
		epsBuffer:    epsBuffer,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data

	randomizer   *mps.MatrixRandomMTGP32
	randomMatrix *mps.Matrix
	epsBuffer    *mtl.Buffer
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	k.randomizer.Encode(b, k.randomMatrix)

	C.vaeSampleForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.epsBuffer.GetID(),
		k.randomMatrix.GetData().GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.vaeSampleBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		k.epsBuffer.GetID(),
		C.uint(k.input.Dims.W),
		C.uint(k.input.Dims.H),
		C.uint(k.output.Dims.W),
		C.uint(k.output.Dims.H),
	)
}
