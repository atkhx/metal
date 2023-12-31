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

void mtlBlitCommandEncoderEndEncoding(void *encID) {
	[(id<MTLBlitCommandEncoder>)encID endEncoding];
}

*/
import "C"
import (
	"unsafe"
)

type NSRange struct {
	Location uint64
	Length   uint64
}

func (r NSRange) C() C.NSRange {
	return C.NSRange(C.NSMakeRange(C.ulong(r.Location), C.ulong(r.Length)))
}

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

func (e *BlitCommandEncoder) EndEncoding() {
	C.mtlBlitCommandEncoderEndEncoding(e.id)
}
