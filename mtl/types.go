package mtl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
*/
import "C"

const (
	dimsWidthIdx  = 0
	dimsHeightIdx = 1
	dimsDepthIdx  = 2
)

func NewMTLSize(dims ...int) MTLSize {
	if len(dims) > 3 {
		panic("to much dimensions")
	}

	allDims := []int{1, 1, 1}
	copy(allDims, dims)

	return MTLSize{
		W: allDims[dimsWidthIdx],
		H: allDims[dimsHeightIdx],
		D: allDims[dimsDepthIdx],
	}
}

type MTLSize struct {
	W, H, D int
}

func MTLSizeFromC(s C.MTLSize) MTLSize {
	return MTLSize{
		W: int(s.width),
		H: int(s.height),
		D: int(s.depth),
	}
}

func (s MTLSize) C() C.MTLSize {
	return C.MTLSizeMake(C.ulong(s.W), C.ulong(s.H), C.ulong(s.D))
}

func (s MTLSize) Length() int {
	return s.W * s.H * s.D
}

func (s MTLSize) GetWHD() (int, int, int) {
	return s.W, s.H, s.D
}

type NSRange struct {
	Location int
	Length   int
}

func NSRangeFromC(r C.NSRange) NSRange {
	return NSRange{
		Location: int(r.location),
		Length:   int(r.length),
	}
}

func (r NSRange) C() C.NSRange {
	return C.NSRange(C.NSMakeRange(C.ulong(r.Location), C.ulong(r.Length)))
}
