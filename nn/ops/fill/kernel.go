package fill

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* fillKernelCreate(void *deviceID, const char *kernelSource) {
    return [[fillKernelImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void fill(
    void *kernelID,
    void *commandBuffer,
    void *dstBuffer,
    float value,
    const uint offset,
    const uint length
) {
    [(__bridge fillKernelImpl*)kernelID fill:(id<MTLCommandBuffer>)commandBuffer
        dstBuffer:(id<MTLBuffer>)dstBuffer
        value:(float)value
        offset:(uint)offset
        length:(uint)length
	];
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

func New(device *mtl.Device) *Kernel {
	cKernelString := C.CString(metalFunctions)
	defer C.free(unsafe.Pointer(cKernelString))
	return &Kernel{
		kernelID: C.fillKernelCreate(device.GetID(), cKernelString),
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
}

func (k *Kernel) Fill(b *mtl.CommandBuffer, target *mtl.Buffer, value float32, offset, length int) {
	C.fill(k.kernelID, b.GetID(), target.GetID(), C.float(value), C.uint(offset*4), C.uint(length*4))
}
