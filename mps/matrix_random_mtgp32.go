package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* mpsMatrixRandomMTGP32Create(void *deviceID, void *distribution, NSUInteger seed) {
    return [[MPSMatrixRandomMTGP32 alloc]
        initWithDevice:(id<MTLDevice>)deviceID
        destinationDataType:MPSDataTypeFloat32
        seed:seed
        distributionDescriptor:(__bridge MPSMatrixRandomDistributionDescriptor*)distribution
    ];
}

void mpsMatrixRandomMTGP32Encode(void *kernelID, void *commandBufferID, void *dstMatrix) {
    [(__bridge MPSMatrixRandomMTGP32*)kernelID
        encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBufferID
        destinationMatrix:(__bridge MPSMatrix*)dstMatrix];
}

void mpsMatrixRandomMTGP32EncodeRelease(void *id) {
	return [(__bridge MPSMatrixRandomMTGP32*)id release];
}

*/
import "C"
import (
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

type MatrixRandomMTGP32 struct {
	id unsafe.Pointer
}

func CreateMatrixRandomMTGP32(
	device *mtl.Device,
	descriptor *MatrixRandomDistributionDescriptor,
	seed uint64,
) *MatrixRandomMTGP32 {
	id := C.mpsMatrixRandomMTGP32Create(device.GetID(), descriptor.GetID(), C.ulong(seed))
	if id == nil {
		panic("MPSMatrixRandomMTGP32: id is empty")
	}
	return &MatrixRandomMTGP32{id: id}
}

func (r *MatrixRandomMTGP32) Release() {
	C.mpsMatrixRandomMTGP32EncodeRelease(r.id)
}

func (r *MatrixRandomMTGP32) GetID() unsafe.Pointer {
	return r.id
}

func (r *MatrixRandomMTGP32) Encode(commandBuffer *mtl.CommandBuffer, matrix *Matrix) {
	C.mpsMatrixRandomMTGP32Encode(r.id, commandBuffer.GetID(), matrix.GetID())
}
