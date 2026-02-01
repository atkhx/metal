package layer

import (
	"fmt"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

// PositionalAdd adds a learnable position embedding tensor (tiled to batch).
type PositionalAdd struct {
	context  int
	features int

	weights   *num.Data
	forUpdate []*num.Data

	provideWeights func(weights *num.Data)
}

func NewPositionalAdd(context, features int, provideWeights func(weights *num.Data)) *PositionalAdd {
	return &PositionalAdd{
		context:        context,
		features:       features,
		provideWeights: provideWeights,
	}
}

func (l *PositionalAdd) Compile(device *proc.Device, input *num.Data) *num.Data {
	if input.Dims.W != l.features || input.Dims.H != l.context {
		panic(fmt.Sprintf("PositionalAdd: expected %dx%d got %dx%d", l.features, l.context, input.Dims.W, input.Dims.H))
	}
	l.weights = device.NewData(mtl.NewMTLSize(l.features, l.context, 1))

	l.forUpdate = []*num.Data{l.weights}
	return device.PositionalAdd(input, l.weights, l.features, l.context)
}

func (l *PositionalAdd) ForUpdate() []*num.Data {
	return l.forUpdate
}

func (l *PositionalAdd) LoadFromProvider() {
	if l.provideWeights != nil {
		l.provideWeights(l.weights)
	}
}
