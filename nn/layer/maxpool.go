package layer

import (
	"fmt"

	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

// MaxPool2D applies 2D max pooling over width/height.
type MaxPool2D struct {
	poolSize int
	stride   int
	padding  int
}

func NewMaxPool2D(poolSize, stride, padding int) *MaxPool2D {
	if poolSize < 1 {
		panic("MaxPool2D: poolSize must be >= 1")
	}
	if stride < 1 {
		stride = poolSize
	}
	if padding < 0 {
		padding = 0
	}
	return &MaxPool2D{
		poolSize: poolSize,
		stride:   stride,
		padding:  padding,
	}
}

func (l *MaxPool2D) Compile(device *proc.Device, input *num.Data) *num.Data {
	if input.Dims.W < l.poolSize || input.Dims.H < l.poolSize {
		panic(fmt.Sprintf("MaxPool2D: poolSize=%d exceeds input %dx%d", l.poolSize, input.Dims.W, input.Dims.H))
	}
	return device.MaxPool2D(input, l.poolSize, l.padding, l.stride)
}
