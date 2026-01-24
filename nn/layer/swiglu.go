package layer

import (
	"encoding/json"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewSwiGLU(featuresCount, hiddenSize int, initWeights initializer.Initializer, provideWeights func(w1, w2, w3 *num.Data)) *SwiGLU {
	if initWeights == nil {
		initWeights = initializer.KaimingNormalReLU
	}
	return &SwiGLU{featuresCount: featuresCount, hiddenSize: hiddenSize, initWeights: initWeights, provideWeights: provideWeights}
}

type SwiGLU struct {
	initWeights    initializer.Initializer
	provideWeights func(w1, w2, w3 *num.Data)

	hiddenSize    int
	featuresCount int

	weights1 *num.Data
	weights2 *num.Data
	weights3 *num.Data

	forUpdate []*num.Data
}

func (l *SwiGLU) Compile(device *proc.Device, input *num.Data) *num.Data {
	inputWidth := input.Dims.W

	l.weights1 = initWeights(device, l.initWeights, mtl.NewMTLSize(l.hiddenSize, inputWidth), inputWidth, l.hiddenSize)
	l.weights2 = initWeights(device, l.initWeights, mtl.NewMTLSize(l.hiddenSize, inputWidth), inputWidth, l.hiddenSize)
	l.weights3 = initWeights(device, l.initWeights, mtl.NewMTLSize(l.featuresCount, l.hiddenSize), l.hiddenSize, l.featuresCount)

	w1Projection := device.MatrixMultiply(input, l.weights1, 1)
	w2Projection := device.MatrixMultiply(input, l.weights2, 1)

	w1SiLU := device.SiLu(w1Projection)

	l.forUpdate = []*num.Data{l.weights1, l.weights2, l.weights3}

	return device.MatrixMultiply(device.MulEqual(w1SiLU, w2Projection), l.weights3, 1)
}

func (l *SwiGLU) ForUpdate() []*num.Data {
	return l.forUpdate
}

type SwiGLUConfig struct {
	Weights1 []float32
	Weights2 []float32
	Weights3 []float32
}

func (l *SwiGLU) MarshalJSON() ([]byte, error) {
	return json.Marshal(SwiGLUConfig{
		Weights1: l.weights1.Data.GetFloats(),
		Weights2: l.weights2.Data.GetFloats(),
		Weights3: l.weights3.Data.GetFloats(),
	})
}

func (l *SwiGLU) UnmarshalJSON(bytes []byte) error {
	config := SwiGLUConfig{
		Weights1: l.weights1.Data.GetFloats(),
		Weights2: l.weights2.Data.GetFloats(),
		Weights3: l.weights3.Data.GetFloats(),
	}
	return json.Unmarshal(bytes, &config)
}

func (l *SwiGLU) LoadFromProvider() {
	l.provideWeights(l.weights1, l.weights2, l.weights3)
}
