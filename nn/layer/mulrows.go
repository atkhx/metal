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

func (l *MulRows) Compile(device *proc.Device, input *num.Data) *num.Data {
	values := make([]float32, l.width)
	for i := range values {
		values[i] = 1
	}
	l.weightObj = device.NewDataWithValues(mtl.NewMTLSize(l.width), values)
	l.forUpdate = []*num.Data{l.weightObj}

	return device.MulRow(input, l.weightObj, l.width)
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
