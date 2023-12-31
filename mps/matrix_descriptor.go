package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void* mpsMatrixDescriptorCreateFloat32(int cols, int rows, int batchSize, int batchStride) {
    return [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows
        columns:cols
        matrices:batchSize
        rowBytes:cols * sizeof(float)
        matrixBytes:batchStride * sizeof(float)
        dataType:MPSDataTypeFloat32];
}

void mpsMatrixDescriptorRelease(void *descriptorID) {
    [(__bridge MPSMatrixDescriptor*)descriptorID release];
}

NSUInteger mpsMatrixDescriptorGetRows(void *descriptorID) {
	return [(__bridge MPSMatrixDescriptor*)descriptorID rows];
}

NSUInteger mpsMatrixDescriptorGetColumns(void *descriptorID) {
	return [(__bridge MPSMatrixDescriptor*)descriptorID columns];
}

NSUInteger mpsMatrixDescriptorGetMatrices(void *descriptorID) {
	return [(__bridge MPSMatrixDescriptor*)descriptorID matrices];
}

MPSDataType mpsMatrixDescriptorGetDataType(void *descriptorID) {
	return [(__bridge MPSMatrixDescriptor*)descriptorID dataType];
}

NSUInteger mpsMatrixDescriptorGetRowBytes(void *descriptorID) {
	return [(__bridge MPSMatrixDescriptor*)descriptorID rowBytes];
}

NSUInteger mpsMatrixDescriptorGetMatrixBytes(void *descriptorID) {
	return [(__bridge MPSMatrixDescriptor*)descriptorID matrixBytes];
}

*/
import "C"
import (
	"unsafe"
)

type MatrixDescriptor struct {
	id unsafe.Pointer
}

func CreateMatrixDescriptorFloat32(cols, rows, batchSize, batchStride int) *MatrixDescriptor {
	return &MatrixDescriptor{id: unsafe.Pointer(C.mpsMatrixDescriptorCreateFloat32(
		C.int(cols),
		C.int(rows),
		C.int(batchSize),
		C.int(batchStride),
	))}
}

func (d *MatrixDescriptor) Release() {
	C.mpsMatrixDescriptorRelease(d.id)
}

func (d *MatrixDescriptor) GetID() unsafe.Pointer {
	return d.id
}

// GetRows The number of rows in a matrix.
func (d *MatrixDescriptor) GetRows() int {
	return int(C.mpsMatrixDescriptorGetRows(d.id))
}

// GetColumns The number of columns in a matrix.
func (d *MatrixDescriptor) GetColumns() int {
	return int(C.mpsMatrixDescriptorGetColumns(d.id))
}

// GetMatrices The number of matrices.
func (d *MatrixDescriptor) GetMatrices() int {
	return int(C.mpsMatrixDescriptorGetMatrices(d.id))
}

// GetDataType The type of the data which makes up the values of the matrix.
func (d *MatrixDescriptor) GetDataType() int {
	return int(C.mpsMatrixDescriptorGetDataType(d.id))
}

// GetRowBytes The stride, in bytes, between corresponding elements of consecutive rows.
// Must be a multiple of the element size.
func (d *MatrixDescriptor) GetRowBytes() int {
	return int(C.mpsMatrixDescriptorGetRowBytes(d.id))
}

// GetMatrixBytes The stride, in bytes, between corresponding elements of consecutive matrices.
// Must be a multiple of rowBytes.
func (d *MatrixDescriptor) GetMatrixBytes() int {
	return int(C.mpsMatrixDescriptorGetMatrixBytes(d.id))
}
