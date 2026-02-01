package layer

import (
	"encoding/json"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewConv(
	filterSize int,
	filtersCount int,
	batchSize int,
	padding int,
	stride int,

	initWeights initializer.Initializer,
	provideWeights func(weights *num.Data),
) *Conv {
	if initWeights == nil {
		initWeights = initializer.KaimingNormalReLU
	}
	if stride < 1 {
		stride = 1
	}
	if batchSize < 1 {
		batchSize = 1
	}
	return &Conv{
		initWeights:    initWeights,
		provideWeights: provideWeights,
		filterSize:     filterSize,
		filtersCount:   filtersCount,
		batchSize:      batchSize,
		padding:        padding,
		stride:         stride,
	}
}

type Conv struct {
	initWeights    initializer.Initializer
	provideWeights func(weights *num.Data)

	filterSize   int
	filtersCount int
	batchSize    int
	padding      int
	stride       int

	weightObj *num.Data
	biasesObj *num.Data

	forUpdate []*num.Data
}

func (l *Conv) GetFiltersCount() int {
	return l.filtersCount
}

func (l *Conv) GetWeights() *num.Data {
	return l.weightObj
}

func (l *Conv) Compile(device *proc.Device, input *num.Data) *num.Data {
	if input.Dims.D%l.batchSize != 0 {
		panic("Conv: input depth must be divisible by batchSize")
	}
	filterDepth := input.Dims.D / l.batchSize

	fanIn := l.filterSize * l.filterSize * filterDepth
	fanOut := l.filterSize * l.filterSize * l.filtersCount

	mFilterSize := mtl.NewMTLSize(l.filterSize, l.filterSize, filterDepth*l.filtersCount)

	l.weightObj = initWeights(device, l.initWeights, mFilterSize, fanIn, fanOut)
	l.biasesObj = device.NewData(mtl.NewMTLSize(1, 1, l.filtersCount))
	l.forUpdate = []*num.Data{l.weightObj, l.biasesObj}

	return device.Conv(input, l.weightObj, l.biasesObj, l.filtersCount, l.batchSize, l.padding, l.stride)
}

func (l *Conv) ForUpdate() []*num.Data {
	return l.forUpdate
}

type convConfig struct {
	Weights []float32
	Bias    []float32
}

func (l *Conv) MarshalJSON() ([]byte, error) {
	return json.Marshal(convConfig{
		Weights: l.weightObj.Data.GetFloats(),
		Bias:    l.biasesObj.Data.GetFloats(),
	})
}

func (l *Conv) UnmarshalJSON(bytes []byte) error {
	cfg := convConfig{
		Weights: l.weightObj.Data.GetFloats(),
		Bias:    l.biasesObj.Data.GetFloats(),
	}
	return json.Unmarshal(bytes, &cfg)
}

func (l *Conv) LoadFromProvider() {
	l.provideWeights(l.weightObj)
}
