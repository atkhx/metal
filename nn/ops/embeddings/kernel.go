package embeddings

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* embeddingsKernelCreate(void *device, const char *kernelSource) {
    return [[EmbeddingsKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
		kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void embeddingsForward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputData,
    void *tokenEmbedding,
    uint featuresCount,
    uint contextLength
) {
    [(__bridge EmbeddingsKernelImpl*)kernel forward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputData:(id<MTLBuffer>)outputData
        tokenEmbedding:(id<MTLBuffer>)tokenEmbedding
        featuresCount:(uint)featuresCount
        contextLength:(uint)contextLength
	];
}

void embeddingsBackward(
    void *kernel,
    void *commandBuffer,
    void *inputData,
    void *outputGrad,
    void *tokenEmbeddingGrad,
	uint featuresCount
) {
    [(__bridge EmbeddingsKernelImpl*)kernel backward:(id<MTLCommandBuffer>)commandBuffer
        inputData:(id<MTLBuffer>)inputData
        outputGrad:(id<MTLBuffer>)outputGrad
        tokenEmbeddingGrad:(id<MTLBuffer>)tokenEmbeddingGrad
        featuresCount:(uint)featuresCount
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
	embeddings *num.Data,
	featuresCount int,
	contextLength int,
) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.embeddingsKernelCreate(device.GetID(), cKernelString),

		device:     device,
		input:      input,
		output:     output,
		embeddings: embeddings,

		featuresCount: featuresCount,
		contextLength: contextLength,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer

	device     *mtl.Device
	input      *num.Data
	output     *num.Data
	embeddings *num.Data

	featuresCount int
	contextLength int
}

func (k *Kernel) Forward(b *mtl.CommandBuffer) {
	C.embeddingsForward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Data.GetID(),
		k.embeddings.Data.GetID(),
		C.uint(k.featuresCount),
		C.uint(k.contextLength),
	)
}

func (k *Kernel) Backward(b *mtl.CommandBuffer) {
	C.embeddingsBackward(
		k.kernelID,
		b.GetID(),
		k.input.Data.GetID(),
		k.output.Grad.GetID(),
		k.embeddings.Grad.GetID(),
		C.uint(k.featuresCount),
	)
}
