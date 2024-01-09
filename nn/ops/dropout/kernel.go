package dropout

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* dropoutKernelCreate(void *device, const char *kernelSource) {
    return [[DropoutKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void dropoutForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *randomData,
	float probability
) {
    [(__bridge DropoutKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
		inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        randomData:(id<MTLBuffer>)randomData
		probability:(float)probability
	];
}

void dropoutBackward(
    void *kernel,
    void *commandBuffer,
    void *inputGrad,
    void *outputGrad,
    void *randomData,
	float probability
) {
    [(__bridge DropoutKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
		inputGrad:(id<MTLBuffer>)inputGrad
        outputGrad:(id<MTLBuffer>)outputGrad
        randomData:(id<MTLBuffer>)randomData
		probability:(float)probability
	];
}

*/
import "C"
import (
	_ "embed"
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
	probability float32,
	seed uint64,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))

	distribution := mps.CreateMatrixRandomDistributionDescriptor(0, 1)
	randomizer := mps.CreateMatrixRandomMTGP32(device, distribution, seed)

	maskBuffer := device.NewBufferEmptyFloatsBuffer(input.Dims.Length(), mtl.ResourceStorageModeShared)
	maskDescriptor := mps.CreateMatrixDescriptorFloat32(input.Dims.W, input.Dims.H, input.Dims.D, input.Dims.W*input.Dims.H)

	maskMatrix := mps.CreateMatrixWithBuffer(maskDescriptor, maskBuffer, 0)

	return &Kernel{
		kernelID: C.dropoutKernelCreate(device.GetID(), cKernelString),

		device: device,
		input:  input,
		output: output,

		randomizer:  randomizer,
		maskMatrix:  maskMatrix,
		probability: probability,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device *mtl.Device
	input  *num.Data
	output *num.Data

	randomizer *mps.MatrixRandomMTGP32
	maskMatrix *mps.Matrix

	probability float32
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	k.randomizer.Encode(b, k.maskMatrix)

	C.dropoutForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.maskMatrix.GetData().GetID(),
		C.float(k.probability),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.dropoutBackward(
		k.kernelID,
		b.GetID(),
		k.input.Grad.GetID(),
		k.output.Grad.GetID(),
		k.maskMatrix.GetData().GetID(),
		C.float(k.probability),
	)
}
