package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewLinearWithImmutableWeights(
	weightObj *num.Data,
) *LinearWithWeights {
	return &LinearWithWeights{
		weightObj: weightObj,
	}
}

type LinearWithWeights struct {
	weightObj *num.Data
}

func (l *LinearWithWeights) Compile(device *proc.Device, input *num.Data) *num.Data {
	return device.MatrixMultiply(input, l.weightObj, 1)
}
