package mtl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
*/
import "C"

type MTLSize struct {
	W, H, D uint64
}

func MTLSizeFromC(s C.MTLSize) MTLSize {
	return MTLSize{
		W: uint64(s.width),
		H: uint64(s.height),
		D: uint64(s.depth),
	}
}

func (s MTLSize) C() C.MTLSize {
	return C.MTLSizeMake(C.ulong(s.W), C.ulong(s.H), C.ulong(s.D))
}

type NSRange struct {
	Location uint64
	Length   uint64
}

func NSRangeFromC(r C.NSRange) NSRange {
	return NSRange{
		Location: uint64(r.location),
		Length:   uint64(r.length),
	}
}

func (r NSRange) C() C.NSRange {
	return C.NSRange(C.NSMakeRange(C.ulong(r.Location), C.ulong(r.Length)))
}
