package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void* mpsMatrixInitWithBuffer(void *bufferID, void *descriptorID, int offset) {
    return [[MPSMatrix alloc]
        initWithBuffer:(id<MTLBuffer>)bufferID
        offset:offset
        descriptor:(__bridge MPSMatrixDescriptor*)descriptorID];
}

void mpsMatrixRelease(void *matrixID) {
    [(__bridge MPSMatrix*)matrixID release];
}

NSUInteger mpsMatrixGetRows(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID rows];
}

NSUInteger mpsMatrixGetColumns(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID columns];
}

NSUInteger mpsMatrixGetMatrices(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID matrices];
}

MPSDataType mpsMatrixGetDataType(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID dataType];
}

NSUInteger mpsMatrixGetRowBytes(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID rowBytes];
}

NSUInteger mpsMatrixGetMatrixBytes(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID matrixBytes];
}

NSUInteger mpsMatrixGetOffset(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID offset];
}

void* mpsMatrixGetData(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID data];
}

void mpsMatrixSynchronizeOnCommandBuffer(void *matrixID, void *commandBufferID) {
	return [(__bridge MPSMatrix*)matrixID synchronizeOnCommandBuffer:(id<MTLCommandBuffer>)commandBufferID];
}

NSUInteger mpsMatrixGetResourceSize(void *matrixID) {
	return [(__bridge MPSMatrix*)matrixID resourceSize];
}

*/
import "C"
import (
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

type Matrix struct {
	id unsafe.Pointer
}

func CreateMatrixWithBuffer(descriptor *MatrixDescriptor, buffer *mtl.Buffer, offset int) *Matrix {
	id := unsafe.Pointer(C.mpsMatrixInitWithBuffer(
		buffer.GetID(),
		descriptor.GetID(),
		C.int(offset),
	))

	if id == nil {
		panic("MPSMatrix: id is empty")
	}

	return &Matrix{id: id}
}

func (m *Matrix) Release() {
	C.mpsMatrixRelease(m.id)
}

func (m *Matrix) GetID() unsafe.Pointer {
	return m.id
}

// GetRows The number of rows in a matrix in the Matrix.
func (m *Matrix) GetRows() int {
	return int(C.mpsMatrixGetRows(m.id))
}

// GetColumns The number of columns in a matrix in the Matrix.
func (m *Matrix) GetColumns() int {
	return int(C.mpsMatrixGetColumns(m.id))
}

// GetMatrices The number of matrices in the Matrix.
func (m *Matrix) GetMatrices() int {
	return int(C.mpsMatrixGetMatrices(m.id))
}

// GetDataType The type of the Matrix data.
func (m *Matrix) GetDataType() uint64 {
	return uint64(C.mpsMatrixGetDataType(m.id))
}

// GetRowBytes The stride, in bytes, between corresponding elements of consecutive rows.
func (m *Matrix) GetRowBytes() int {
	return int(C.mpsMatrixGetRowBytes(m.id))
}

// GetMatrixBytes The stride, in bytes, between corresponding elements of consecutive matrices.
func (m *Matrix) GetMatrixBytes() int {
	return int(C.mpsMatrixGetMatrixBytes(m.id))
}

// GetOffset Byte-offset to the buffer where the matrix data begins - see @ref initWithBuffer: offset: descriptor
func (m *Matrix) GetOffset() int {
	return int(C.mpsMatrixGetOffset(m.id))
}

// GetData An MTLBuffer to store the data.
func (m *Matrix) GetData() *mtl.Buffer {
	return mtl.CreateBuffer(unsafe.Pointer(C.mpsMatrixGetData(m.id)))
}

// SynchronizeOnCommandBuffer Flush the underlying MTLBuffer from the device's caches, and invalidate any CPU caches if needed.
// This will call MTLBlitEncoder.synchronizeResource on the matrix's MTLBuffer, if any.
// This is necessary for all storageModeManaged resources. For other resources, including temporary
// resources (these are all storageModePrivate), and buffers that have not yet been allocated, nothing is done.
// It is more efficient to use this method than to attempt to do this yourself with the data property.
func (m *Matrix) SynchronizeOnCommandBuffer(commandBuffer *mtl.CommandBuffer) {
	C.mpsMatrixSynchronizeOnCommandBuffer(m.id, commandBuffer.GetID())
}

// GetResourceSize Get the number of bytes used to allocate underlying MTLResources.
func (m *Matrix) GetResourceSize() int {
	return int(C.mpsMatrixGetResourceSize(m.id))
}
