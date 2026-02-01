package mtl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>

void* mtlBufferGetContentsPointer(void *bufferID) {
    return [(id<MTLBuffer>)bufferID contents];
}

void mtlBufferRelease(void *bufferID) {
    [(id<MTLBuffer>)bufferID release];
}

void* mtlBufferGetContents(void *bufferID) {
    return [(id<MTLBuffer>)bufferID contents];
}

NSUInteger mtlBufferGetLengthBytes(void *bufferID) {
    return [(id<MTLBuffer>)bufferID length];
}

NSUInteger mtlBufferGetLengthFloats(void *bufferID) {
    return [(id<MTLBuffer>)bufferID length] / sizeof(float);
}

*/
import "C"
import (
	"unsafe"
)

type Buffer struct {
	id unsafe.Pointer
}

func CreateBuffer(id unsafe.Pointer) *Buffer {
	if id == nil {
		panic("MTLBuffer: id is empty")
	}
	return &Buffer{id: id}
}

func (b *Buffer) Release() {
	C.mtlBufferRelease(b.id)
}

func (b *Buffer) GetID() unsafe.Pointer {
	return b.id
}

func (b *Buffer) GetLengthBytes() uint64 {
	return uint64(C.mtlBufferGetLengthBytes(b.id))
}

func (b *Buffer) GetLengthFloats() uint64 {
	return uint64(C.mtlBufferGetLengthFloats(b.id))
}

func (b *Buffer) GetContents() unsafe.Pointer {
	return C.mtlBufferGetContents(b.id)
}

func (b *Buffer) GetBytes() []byte {
	return unsafe.Slice((*byte)(b.GetContents()), int(b.GetLengthBytes()))
}

func (b *Buffer) GetFloats() []float32 {
	return unsafe.Slice((*float32)(b.GetContents()), int(b.GetLengthFloats()))
}
