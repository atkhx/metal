package model

import (
	"fmt"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

// LayerNormAffine applies LayerNorm followed by affine (gamma/beta).
type LayerNormAffine struct {
	width       int
	initWeights initializer.Initializer

	gamma     *num.Data
	beta      *num.Data
	skip      bool
	forUpdate []*num.Data
}

func NewLayerNormAffine(width int, init initializer.Initializer) *LayerNormAffine {
	if init == nil {
		init = initializer.InitWeightFixed{NormK: 1}
	}
	return &LayerNormAffine{width: width, initWeights: init}
}

func (l *LayerNormAffine) Skip(value bool) {
	l.skip = value
}

func (l *LayerNormAffine) Compile(device *proc.Device, input *num.Data) *num.Data {
	if input.Dims.W != l.width {
		panic(fmt.Sprintf("LayerNormAffine: expected width=%d got %d", l.width, input.Dims.W))
	}
	if l.gamma == nil {
		values := make([]float32, l.width)
		for i := range values {
			values[i] = 1
		}
		l.gamma = device.NewDataWithValues(mtl.NewMTLSize(l.width), values)
	}
	if l.beta == nil {
		l.beta = device.NewData(mtl.NewMTLSize(l.width))
	}
	l.forUpdate = []*num.Data{l.gamma, l.beta}

	if l.skip {
		return input
	}

	out := input
	//out = device.RMSNorm(out, l.width)
	out = device.LayerNorm(out, l.width)
	out = device.MulRow(out, l.gamma, l.width)
	out = device.AddRow(out, l.beta, l.width)
	return out
}

func (l *LayerNormAffine) ForUpdate() []*num.Data {
	return l.forUpdate
}
