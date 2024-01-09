package layer

import (
	"encoding/json"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewMulRows(
	width int,
	initWeights initializer.Initializer,
	provideWeights func(weights *num.Data),
) *MulRows {
	return &MulRows{width: width, initWeights: initWeights, provideWeights: provideWeights}
}

type MulRows struct {
	initWeights    initializer.Initializer
	provideWeights func(weights *num.Data)

	width int

	weightObj *num.Data
	forUpdate []*num.Data
}

func (l *MulRows) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(inputs.Dims.Length())

	l.weightObj = device.NewDataRandNormWeighted(mtl.NewMTLSize(l.width), weightK)
	l.forUpdate = []*num.Data{l.weightObj}

	return device.MulRow(inputs, l.weightObj, l.width)
}

func (l *MulRows) ForUpdate() []*num.Data {
	return l.forUpdate
}

type mulRowsConfig struct {
	Weights []float32
}

func (l *MulRows) MarshalJSON() ([]byte, error) {
	return json.Marshal(mulRowsConfig{
		Weights: l.weightObj.Data.GetFloats(),
	})
}

func (l *MulRows) UnmarshalJSON(bytes []byte) error {
	config := mulRowsConfig{
		Weights: l.weightObj.Data.GetFloats(),
	}
	return json.Unmarshal(bytes, &config)
}

func (l *MulRows) LoadFromProvider() {
	l.provideWeights(l.weightObj)
}
