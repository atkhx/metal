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
	provideWeights func(weights, bias *num.Data),
) *Linear {
	if initWeights == nil {
		initWeights = initializer.XavierNormalLinear
	}
	return &Linear{
		featuresCount:  featuresCount,
		initWeights:    initializer.DefaultOnNil(initWeights, initializer.XavierNormalLinear),
		withBias:       withBias,
		provideWeights: provideWeights,
	}
}

type Linear struct {
	initWeights    initializer.Initializer
	provideWeights func(weights, bias *num.Data)
	featuresCount  int

	withBias  bool
	weightObj *num.Data
	biasesObj *num.Data

	output    *num.Data
	forUpdate []*num.Data
}

func (l *Linear) GetWeights() *num.Data {
	return l.weightObj
}

func (l *Linear) Compile(device *proc.Device, input *num.Data) *num.Data {
	inputWidth := input.Dims.W
	outputDims := mtl.NewMTLSize(l.featuresCount, inputWidth)
	l.weightObj = initWeights(device, l.initWeights, outputDims, inputWidth, l.featuresCount)
	l.forUpdate = []*num.Data{l.weightObj}

	result := device.MatrixMultiply(input, l.weightObj, 1)

	if l.withBias {
		l.biasesObj = device.NewData(mtl.NewMTLSize(l.featuresCount))
		l.forUpdate = append(l.forUpdate, l.biasesObj)

		result = device.AddRow(result, l.biasesObj, l.featuresCount)
	}

	l.output = result
	return result
}

func (l *Linear) GetOutput() *num.Data {
	return l.output
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
	if l.provideWeights != nil {
		l.provideWeights(l.weightObj, l.biasesObj)
	}
}
