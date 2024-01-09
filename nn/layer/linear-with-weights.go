package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewLinearWithWeights(
	weightObj *num.Data,
) *LinearWithWeights {
	return &LinearWithWeights{
		weightObj: weightObj,
	}
}

type LinearWithWeights struct {
	weightObj *num.Data
}

func (l *LinearWithWeights) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	return device.MatrixMultiply(inputs, l.weightObj, 1)
}
