package mtl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void mtlBlitCommandEncoderFillBuffer(void *encID, void *bufferID, NSRange range, uint8_t value) {
	[(id<MTLBlitCommandEncoder>)encID
		fillBuffer:(id<MTLBuffer>)bufferID
		range:range
		value:value
	];
}

void mtlBlitCommandEncoderCopyBuffer(void *encID, void *srcBufferID, NSUInteger srcOffset, void *dstBufferID, NSUInteger dstOffset, NSUInteger size) {
	[(id<MTLBlitCommandEncoder>)encID
		copyFromBuffer:(id<MTLBuffer>)srcBufferID
		sourceOffset:srcOffset
		toBuffer:(id<MTLBuffer>)dstBufferID
		destinationOffset:dstOffset
		size:size
	];
}

void mtlBlitCommandEncoderEndEncoding(void *encID) {
	[(id<MTLBlitCommandEncoder>)encID endEncoding];
}

*/
import "C"
import (
	"unsafe"
)

type BlitCommandEncoder struct {
	id unsafe.Pointer
}

func CreateBlitCommandEncoder(id unsafe.Pointer) *BlitCommandEncoder {
	if id == nil {
		panic("MTLBlitCommandEncoder: id is empty")
	}

	return &BlitCommandEncoder{id: id}
}

func (e *BlitCommandEncoder) GetID() unsafe.Pointer {
	return e.id
}

func (e *BlitCommandEncoder) FillBuffer(buffer *Buffer, nsRange NSRange, value byte) {
	C.mtlBlitCommandEncoderFillBuffer(e.id, buffer.GetID(), nsRange.C(), C.uint8_t(value))
}

func (e *BlitCommandEncoder) CopyBuffer(src *Buffer, srcOffset uint64, dst *Buffer, dstOffset uint64, size uint64) {
	C.mtlBlitCommandEncoderCopyBuffer(
		e.id,
		src.GetID(),
		C.NSUInteger(srcOffset),
		dst.GetID(),
		C.NSUInteger(dstOffset),
		C.NSUInteger(size),
	)
}

func (e *BlitCommandEncoder) EndEncoding() {
	C.mtlBlitCommandEncoderEndEncoding(e.id)
}
