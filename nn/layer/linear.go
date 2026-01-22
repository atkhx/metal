package layer

import (
	"encoding/json"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewLinear(
	featuresCount int,
	initWeights initializer.Initializer,
	withBias bool,
	provideWeights func(weights *num.Data),
) *Linear {
	return &Linear{
		featuresCount:  featuresCount,
		initWeights:    initWeights,
		withBias:       withBias,
		provideWeights: provideWeights,
	}
}

type Linear struct {
	initWeights    initializer.Initializer
	provideWeights func(weights *num.Data)
	featuresCount  int

	withBias  bool
	weightObj *num.Data
	biasesObj *num.Data

	forUpdate []*num.Data
}

func (l *Linear) GetWeights() *num.Data {
	return l.weightObj
}

func (l *Linear) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	inputWidth := device.GetDataDims(inputs).W
	outputDims := mtl.NewMTLSize(l.featuresCount, inputWidth)
	weightK := l.initWeights.GetNormK(inputWidth, l.featuresCount)

	l.weightObj = device.NewDataRandNormWeighted(outputDims, weightK)
	l.forUpdate = []*num.Data{l.weightObj}

	result := device.MatrixMultiply(inputs, l.weightObj, 1)

	if l.withBias {
		l.biasesObj = device.NewData(mtl.NewMTLSize(l.featuresCount))
		l.forUpdate = append(l.forUpdate, l.biasesObj)

		result = device.AddRow(result, l.biasesObj, l.featuresCount)
	}

	return result
}

func (l *Linear) ForUpdate() []*num.Data {
	return l.forUpdate
}

type linearConfig struct {
	WithBias bool
	Weights  []float32
	Bias     []float32
}

func (l *Linear) MarshalJSON() ([]byte, error) {
	cfg := linearConfig{
		Weights:  l.weightObj.Data.GetFloats(),
		WithBias: l.withBias,
	}
	if l.biasesObj != nil {
		cfg.Bias = l.biasesObj.Data.GetFloats()
	}
	return json.Marshal(cfg)
}

func (l *Linear) UnmarshalJSON(bytes []byte) error {
	cfg := linearConfig{
		Weights:  l.weightObj.Data.GetFloats(),
		WithBias: l.withBias,
	}
	if l.biasesObj != nil {
		cfg.Bias = l.biasesObj.Data.GetFloats()
	}
	return json.Unmarshal(bytes, &cfg)
}

func (l *Linear) LoadFromProvider() {
	l.provideWeights(l.weightObj)
}
