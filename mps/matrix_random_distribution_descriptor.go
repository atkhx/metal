package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Metal/Metal.h>

void* mpsMatrixRandomDistributionDescriptorCreate(float min, float max) {
    return [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:min maximum:max];
}

void mpsMatrixRandomDistributionDescriptorRelease(void *id) {
	return [(__bridge MPSMatrixRandomDistributionDescriptor*)id release];
}
*/
import "C"
import "unsafe"

type MatrixRandomDistributionDescriptor struct {
	id unsafe.Pointer
}

func CreateMatrixRandomDistributionDescriptor(min, max float32) *MatrixRandomDistributionDescriptor {
	id := C.mpsMatrixRandomDistributionDescriptorCreate(C.float(min), C.float(max))
	if id == nil {
		panic("MPSMatrixRandomDistributionDescriptor: id is empty")
	}
	return &MatrixRandomDistributionDescriptor{id: id}
}

func (d *MatrixRandomDistributionDescriptor) Release() {
	C.mpsMatrixRandomDistributionDescriptorRelease(d.id)
}

func (d *MatrixRandomDistributionDescriptor) GetID() unsafe.Pointer {
	return d.id
}
