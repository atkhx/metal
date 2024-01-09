package adamw

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include "kernel.h"

void* adamWCreate(void *deviceID, const char *kernelSource) {
    return [[MPSAdamWImpl alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        kernelSource:[NSString stringWithUTF8String:kernelSource]];
}

void adamWUpdate(
    void *kernelID,
    void *commandBufferID,
    void *dataBufferID,
    void *gradBufferID,
    void *mBufferID,
    void *vBufferID,
    float beta1,
    float beta2,
    float beta1powIterationLR,
    float beta2powIteration
) {
    [(__bridge MPSAdamWImpl*)kernelID updateWithAdam:(id<MTLCommandBuffer>)commandBufferID
        dataBuffer:(id<MTLBuffer>)dataBufferID
        gradBuffer:(id<MTLBuffer>)gradBufferID
        mBuffer:(id<MTLBuffer>)mBufferID
        vBuffer:(id<MTLBuffer>)vBufferID
        beta1:beta1
        beta2:beta2
        beta1powIterationLR:beta1powIterationLR
        beta2powIteration:beta2powIteration];
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
		kernelID: C.adamWCreate(device.GetID(), cKernelString),
		device:   device,
	}
}

type Kernel struct {
	kernelID unsafe.Pointer
	device   *mtl.Device
}

func (k *Kernel) UpdateWithAdam(
	commandBuffer *mtl.CommandBuffer,
	dataBuffer *mtl.Buffer,
	gradBuffer *mtl.Buffer,
	mBuffer *mtl.Buffer,
	vBuffer *mtl.Buffer,
	beta1 float32,
	beta2 float32,
	beta1powIterationLR float32,
	beta2powIteration float32,
) {
	C.adamWUpdate(
		k.kernelID,
		commandBuffer.GetID(),

		dataBuffer.GetID(),
		gradBuffer.GetID(),
		mBuffer.GetID(),
		vBuffer.GetID(),

		C.float(beta1),
		C.float(beta2),
		C.float(beta1powIterationLR),
		C.float(beta2powIteration),
	)
}
