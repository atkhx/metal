package mpspack

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* mpsPackKernelCreate(void *device, const char *kernelSource) {
    return [[MPSPackKernelImpl alloc] initWithDevice:(id<MTLDevice>)device
                                        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void mpsPackEncode(
    void *kernel,
    void *commandBuffer,
    void *inputBuffer,
    void *outputTexture,
    uint width,
    uint height,
    uint channels,
    uint batchSize,
    uint slices
) {
    [(__bridge MPSPackKernelImpl*)kernel pack:(id<MTLCommandBuffer>)commandBuffer
                                         input:(id<MTLBuffer>)inputBuffer
                                        output:(id<MTLTexture>)outputTexture
                                          width:width
                                         height:height
                                       channels:channels
                                      batchSize:batchSize
                                         slices:slices];
}

void mpsUnpackEncode(
    void *kernel,
    void *commandBuffer,
    void *inputTexture,
    void *outputBuffer,
    uint width,
    uint height,
    uint channels,
    uint batchSize,
    uint slices
) {
    [(__bridge MPSPackKernelImpl*)kernel unpack:(id<MTLCommandBuffer>)commandBuffer
                                          input:(id<MTLTexture>)inputTexture
                                         output:(id<MTLBuffer>)outputBuffer
                                           width:width
                                          height:height
                                        channels:channels
                                       batchSize:batchSize
                                          slices:slices];
}

*/
import "C"
import (
	_ "embed"
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

//go:embed kernel.metal
var metalFunctions string

type Kernel struct {
	id unsafe.Pointer
}

func New(device *mtl.Device) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		id: C.mpsPackKernelCreate(device.GetID(), cKernelString),
	}
}

func (k *Kernel) Pack(b *mtl.CommandBuffer, input *mtl.Buffer, outputTexture unsafe.Pointer, width, height, channels, batchSize int) {
	slices := (channels + 3) / 4
	C.mpsPackEncode(
		k.id,
		b.GetID(),
		input.GetID(),
		outputTexture,
		C.uint(width),
		C.uint(height),
		C.uint(channels),
		C.uint(batchSize),
		C.uint(slices),
	)
}

func (k *Kernel) Unpack(b *mtl.CommandBuffer, inputTexture unsafe.Pointer, output *mtl.Buffer, width, height, channels, batchSize int) {
	slices := (channels + 3) / 4
	C.mpsUnpackEncode(
		k.id,
		b.GetID(),
		inputTexture,
		output.GetID(),
		C.uint(width),
		C.uint(height),
		C.uint(channels),
		C.uint(batchSize),
		C.uint(slices),
	)
}
