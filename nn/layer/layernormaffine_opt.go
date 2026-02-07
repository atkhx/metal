package layer

import (
	"fmt"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

// LayerNormAffineOpt applies LayerNorm (optimized kernel) followed by affine (gamma/beta).
type LayerNormAffineOpt struct {
	width int
	eps   float32

	gamma     *num.Data
	beta      *num.Data
	forUpdate []*num.Data

	provideWeights func(gamma, beta *num.Data)
}

func NewLayerNormAffineOpt(
	width int,
	eps float32,
	provideWeights func(gamma, beta *num.Data),
) *LayerNormAffineOpt {
	if eps <= 0 {
		eps = 1e-5
	}
	return &LayerNormAffineOpt{
		width:          width,
		eps:            eps,
		provideWeights: provideWeights,
	}
}

func (l *LayerNormAffineOpt) Compile(device *proc.Device, input *num.Data) *num.Data {
	if input.Dims.W != l.width {
		panic(fmt.Sprintf("LayerNormAffineOpt: expected width=%d got %d", l.width, input.Dims.W))
	}
	values := make([]float32, l.width)
	for i := range values {
		values[i] = 1
	}
	l.gamma = device.NewDataWithValues(mtl.NewMTLSize(l.width), values)
	l.beta = device.NewData(mtl.NewMTLSize(l.width))
	l.forUpdate = []*num.Data{l.gamma, l.beta}

	out := input
	out = device.LayerNormOpt(out, l.width, l.eps)
	out = device.MulRow(out, l.gamma, l.width)
	out = device.AddRow(out, l.beta, l.width)
	return out
}

func (l *LayerNormAffineOpt) ForUpdate() []*num.Data {
	return l.forUpdate
}

func (l *LayerNormAffineOpt) LoadFromProvider() {
	if l.provideWeights != nil {
		l.provideWeights(l.gamma, l.beta)
	}
}
