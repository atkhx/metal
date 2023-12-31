package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* mpsMatrixMultiplicationCreate(
    void *deviceID,

    int resultRows,
    int resultColumns,
    int interiorColumns,

    float alpha,
    float beta,

    bool transposeLeft,
    bool transposeRight
) {
    return [[MPSMatrixMultiplication alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        transposeLeft:transposeLeft
        transposeRight:transposeRight
        resultRows:resultRows
        resultColumns:resultColumns
        interiorColumns:interiorColumns
        alpha:alpha
        beta:beta];
}

void mpsMatrixMultiplicationEncode(
    void *commandBufferID,
    void *kernelID,
    void *matrixAID,
    void *matrixBID,
    void *matrixCID
) {
    [(__bridge MPSMatrixMultiplication*)kernelID encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        leftMatrix:(__bridge MPSMatrix*)matrixAID
        rightMatrix:(__bridge MPSMatrix*)matrixBID
        resultMatrix:(__bridge MPSMatrix*)matrixCID];
}

void mpsMatrixMultiplicationRelease(void *kernelID) {
    [(__bridge MPSMatrixMultiplication*)kernelID release];
}

*/
import "C"
import (
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

type MatrixMultiplicationKernel struct {
	id unsafe.Pointer
}

func CreateMatrixMultiplicationKernel(
	device *mtl.Device,

	resultRows,
	resultColumns,
	interiorColumns int,

	alpha,
	beta float32,

	transposeLeft,
	transposeRight bool,

) *MatrixMultiplicationKernel {
	id := unsafe.Pointer(C.mpsMatrixMultiplicationCreate(
		device.GetID(),

		C.int(resultRows),
		C.int(resultColumns),
		C.int(interiorColumns),

		C.float(alpha),
		C.float(beta),

		C.bool(transposeLeft),
		C.bool(transposeRight),
	))

	if id == nil {
		panic("MPSMatrixMultiplication: id is empty")
	}

	return &MatrixMultiplicationKernel{id: id}
}

func (k *MatrixMultiplicationKernel) Release() {
	C.mpsMatrixMultiplicationRelease(k.id)
}

func (k *MatrixMultiplicationKernel) Encode(
	commandBuffer *mtl.CommandBuffer,
	aMatrix *Matrix,
	bMatrix *Matrix,
	cMatrix *Matrix,
) {
	C.mpsMatrixMultiplicationEncode(
		commandBuffer.GetID(),
		k.id,
		aMatrix.GetID(),
		bMatrix.GetID(),
		cMatrix.GetID(),
	)
}
