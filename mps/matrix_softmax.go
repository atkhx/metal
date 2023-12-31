package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

// MPSMatrixSoftMax

void* mpsMatrixSoftMaxCreate(void *deviceID) {
    return [[MPSMatrixSoftMax alloc] initWithDevice:(id<MTLDevice>)deviceID];
}

void mpsMatrixSoftMaxRelease(void *id) {
	return [(__bridge MPSMatrixSoftMax*)id release];
}

void mpsMatrixSoftMaxEncode(
    void *commandBufferID,
    void *kernelID,
    void *inputMatrix,
    void *resultMatrix
) {
    [(__bridge MPSMatrixSoftMax*)kernelID encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        inputMatrix:(__bridge MPSMatrix*)inputMatrix
        resultMatrix:(__bridge MPSMatrix*)resultMatrix];
}

*/
import "C"
import (
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

type MatrixSoftMaxKernel struct {
	id unsafe.Pointer
}

func CreateMatrixSoftMaxKernel(device *mtl.Device) *MatrixSoftMaxKernel {
	id := unsafe.Pointer(C.mpsMatrixSoftMaxCreate(device.GetID()))
	if id == nil {
		panic("MPSMatrixSoftMax: id is empty")
	}

	return &MatrixSoftMaxKernel{id: id}
}

func (k *MatrixSoftMaxKernel) Release() {
	C.mpsMatrixSoftMaxRelease(k.id)
}

func (k *MatrixSoftMaxKernel) GetID() unsafe.Pointer {
	return k.id
}

func (k *MatrixSoftMaxKernel) Encode(commandBuffer *mtl.CommandBuffer, input, output *Matrix) {
	C.mpsMatrixSoftMaxEncode(commandBuffer.GetID(), k.id, input.GetID(), output.GetID())
}
