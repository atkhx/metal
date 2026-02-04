package layer

import (
	"fmt"

	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

// UpSample2D applies nearest-neighbor upsampling over width/height.
type UpSample2D struct {
	scale int
}

func NewUpSample2D(scale int) *UpSample2D {
	if scale < 1 {
		panic("UpSample2D: scale must be >= 1")
	}
	return &UpSample2D{scale: scale}
}

func (l *UpSample2D) Compile(device *proc.Device, input *num.Data) *num.Data {
	if input.Dims.W < 1 || input.Dims.H < 1 {
		panic(fmt.Sprintf("UpSample2D: invalid input %dx%d", input.Dims.W, input.Dims.H))
	}
	return device.UpSample2D(input, l.scale)
}
