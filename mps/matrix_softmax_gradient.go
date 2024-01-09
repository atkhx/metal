package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void* mpsMatrixSoftMaxGradientCreate(void *deviceID) {
    return [[MPSMatrixSoftMaxGradient alloc] initWithDevice:(id<MTLDevice>)deviceID];
}

void mpsMatrixSoftMaxGradientRelease(void *id) {
	return [(__bridge MPSMatrixSoftMaxGradient*)id release];
}

void mpsMatrixSoftMaxGradientEncode(
    void *commandBufferID,
    void *kernelID,
    void *gradientMatrix,
    void *forwardOutputMatrix,
    void *resultMatrix
) {
    [(__bridge MPSMatrixSoftMaxGradient*)kernelID encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        gradientMatrix:(__bridge MPSMatrix*)gradientMatrix
        forwardOutputMatrix:(__bridge MPSMatrix*)forwardOutputMatrix
        resultMatrix:(__bridge MPSMatrix*)resultMatrix];
}

*/
import "C"
import (
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

type MatrixSoftMaxGradientKernel struct {
	id unsafe.Pointer
}

func CreateMatrixSoftMaxGradientKernel(device *mtl.Device) *MatrixSoftMaxGradientKernel {
	id := unsafe.Pointer(C.mpsMatrixSoftMaxGradientCreate(device.GetID()))
	if id == nil {
		panic("MPSMatrixSoftMaxGradient: id is empty")
	}
	return &MatrixSoftMaxGradientKernel{id: id}
}

func (k *MatrixSoftMaxGradientKernel) Release() {
	C.mpsMatrixSoftMaxGradientRelease(k.id)
}

func (k *MatrixSoftMaxGradientKernel) Encode(
	commandBuffer *mtl.CommandBuffer,
	gradientMatrix *Matrix,
	forwardOutputMatrix *Matrix,
	resultMatrix *Matrix,
) {
	C.mpsMatrixSoftMaxGradientEncode(
		commandBuffer.GetID(),
		k.id,
		gradientMatrix.GetID(),
		forwardOutputMatrix.GetID(),
		resultMatrix.GetID(),
	)
}
