package layer

import (
	"encoding/json"
	"math"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewSAMultiHeadWithBias(
	featuresCount int,
	headSize int,
	headsCount int,
	contextLength int,
	useRoPE bool,
	initWeights initializer.Initializer,
	provideWeights func(qw, kw, vw, qb, kb, vb *num.Data),
) *SAMultiHeadWithBias {
	return &SAMultiHeadWithBias{
		initWeights:    initializer.DefaultOnNil(initWeights, initializer.XavierNormalLinear),
		featuresCount:  featuresCount,
		contextLength:  contextLength,
		headsCount:     headsCount,
		headSize:       headSize,
		provideWeights: provideWeights,
		useRoPE:        useRoPE,
	}
}

type SAMultiHeadWithBias struct {
	qryWeights *num.Data
	keyWeights *num.Data
	valWeights *num.Data
	qryBias    *num.Data
	keyBias    *num.Data
	valBias    *num.Data

	forUpdate []*num.Data

	initWeights    initializer.Initializer
	provideWeights func(qw, kw, vw, qb, kb, vb *num.Data)

	featuresCount int
	contextLength int
	headsCount    int
	headSize      int
	useRoPE       bool
}

func (l *SAMultiHeadWithBias) Compile(device *proc.Device, input *num.Data) *num.Data {
	fanIn := input.Dims.Length()
	fanOut := l.featuresCount
	batchSize := input.Dims.D

	l.qryWeights = initWeights(device, l.initWeights, mtl.NewMTLSize(l.featuresCount, l.featuresCount), fanIn, fanOut)
	l.keyWeights = initWeights(device, l.initWeights, mtl.NewMTLSize(l.featuresCount, l.featuresCount), fanIn, fanOut)
	l.valWeights = initWeights(device, l.initWeights, mtl.NewMTLSize(l.featuresCount, l.featuresCount), fanIn, fanOut)

	l.qryBias = device.NewData(mtl.NewMTLSize(l.featuresCount))
	l.keyBias = device.NewData(mtl.NewMTLSize(l.featuresCount))
	l.valBias = device.NewData(mtl.NewMTLSize(l.featuresCount))

	l.forUpdate = []*num.Data{l.qryWeights, l.keyWeights, l.valWeights, l.qryBias, l.keyBias, l.valBias}

	bx := input
	bx = device.Transpose(bx)

	// Extract qkv-objects
	qryObject := device.MatrixMultiply(l.qryWeights, bx, 1)
	keyObject := device.MatrixMultiply(l.keyWeights, bx, 1)
	valObject := device.MatrixMultiply(l.valWeights, bx, 1)

	// Add biases by rows (features)
	qryObject = device.AddCol(qryObject, l.qryBias, qryObject.Dims.W, qryObject.Dims.H)
	keyObject = device.AddCol(keyObject, l.keyBias, keyObject.Dims.W, keyObject.Dims.H)
	valObject = device.AddCol(valObject, l.valBias, valObject.Dims.W, valObject.Dims.H)

	// Apply RoPE if enabled
	if l.useRoPE {
		qryObject = device.RopeCols(qryObject, l.featuresCount, l.headSize, l.contextLength)
		keyObject = device.RopeCols(keyObject, l.featuresCount, l.headSize, l.contextLength)
	}

	// Reshape qkv-objects
	reshapeToDims := mtl.NewMTLSize(l.contextLength, l.headSize, l.headsCount*batchSize)

	qryObject = device.Reshape(qryObject, reshapeToDims)
	keyObject = device.Reshape(keyObject, reshapeToDims)
	valObject = device.Reshape(valObject, reshapeToDims)

	// Transpose q and v
	qryObject = device.Transpose(qryObject)
	valObject = device.Transpose(valObject)

	// Extract weiObject
	k := float32(math.Pow(float64(l.headSize), -0.5)) // 0.125
	weiObject := device.MatrixMultiply(qryObject, keyObject, k)

	// Apply triangle lower softmax
	weiSoftmax := device.TriangleLowerSoftmax(weiObject)

	// Get MHA-output objects
	bx = device.MatrixMultiply(weiSoftmax, valObject, 1) // bx - horizontal stacked

	// Transpose output before reshape
	bx = device.Transpose(bx) // bx - vertical stacked

	// Reshape output back to big matrix (instead of concatenation)
	bx = device.Reshape(bx, mtl.NewMTLSize(l.contextLength, l.featuresCount, batchSize)) // bx - vertical

	bx = device.Transpose(bx) // bx - horizontal
	return bx
}

func (l *SAMultiHeadWithBias) ForUpdate() []*num.Data {
	return l.forUpdate
}

func (l *SAMultiHeadWithBias) LoadFromProvider() {
	if l.provideWeights != nil {
		l.provideWeights(l.qryWeights, l.keyWeights, l.valWeights, l.qryBias, l.keyBias, l.valBias)
	}
}

type saMultiHeadWithBiasConfig struct {
	QryWeights []float32
	KeyWeights []float32
	ValWeights []float32

	QryBias []float32
	KeyBias []float32
	ValBias []float32
}

func (l *SAMultiHeadWithBias) MarshalJSON() ([]byte, error) {
	config := saMultiHeadWithBiasConfig{
		QryWeights: l.qryWeights.Data.GetFloats(),
		KeyWeights: l.keyWeights.Data.GetFloats(),
		ValWeights: l.valWeights.Data.GetFloats(),

		QryBias: l.qryBias.Data.GetFloats(),
		KeyBias: l.keyBias.Data.GetFloats(),
		ValBias: l.valBias.Data.GetFloats(),
	}
	return json.Marshal(config)
}

func (l *SAMultiHeadWithBias) UnmarshalJSON(bytes []byte) error {
	config := saMultiHeadWithBiasConfig{
		QryWeights: l.qryWeights.Data.GetFloats(),
		KeyWeights: l.keyWeights.Data.GetFloats(),
		ValWeights: l.valWeights.Data.GetFloats(),

		QryBias: l.qryBias.Data.GetFloats(),
		KeyBias: l.keyBias.Data.GetFloats(),
		ValBias: l.valBias.Data.GetFloats(),
	}
	return json.Unmarshal(bytes, &config)
}
