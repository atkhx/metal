package layer

import (
	"encoding/json"
	"math"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewSAMultiHead(
	featuresCount int,
	headSize int,
	headsCount int,
	contextLength int,
	initWeights initializer.Initializer,
	provideWeights func(qw, kw, vw *num.Data),
) *SAMultiHead {
	return &SAMultiHead{
		initWeights:    initWeights,
		featuresCount:  featuresCount,
		contextLength:  contextLength,
		headsCount:     headsCount,
		headSize:       headSize,
		provideWeights: provideWeights,
	}
}

type SAMultiHead struct {
	QryWeights *num.Data
	KeyWeights *num.Data
	ValWeights *num.Data

	forUpdate []*num.Data

	initWeights    initializer.Initializer
	provideWeights func(qw, kw, vw *num.Data)

	featuresCount int
	contextLength int
	headsCount    int
	headSize      int
}

func (l *SAMultiHead) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	weightK := l.initWeights.GetNormK(inputs.Dims.Length())
	batchSize := inputs.Dims.D

	l.QryWeights = device.NewDataRandNormWeighted(mtl.NewMTLSize(l.featuresCount, l.featuresCount), weightK)
	l.KeyWeights = device.NewDataRandNormWeighted(mtl.NewMTLSize(l.featuresCount, l.featuresCount), weightK)
	l.ValWeights = device.NewDataRandNormWeighted(mtl.NewMTLSize(l.featuresCount, l.featuresCount), weightK)

	l.forUpdate = []*num.Data{l.QryWeights, l.KeyWeights, l.ValWeights}

	bx := device.Transpose(inputs) // bx - vertical

	// Extract qkv-objects
	qryObject := device.MatrixMultiply(l.QryWeights, bx, 1)
	keyObject := device.MatrixMultiply(l.KeyWeights, bx, 1)
	valObject := device.MatrixMultiply(l.ValWeights, bx, 1)

	// Apply RoPE
	qryObject = device.RopeCols(qryObject, l.featuresCount, l.headSize, l.contextLength)
	keyObject = device.RopeCols(keyObject, l.featuresCount, l.headSize, l.contextLength)

	// Reshape qkv-objects
	reshapeToDims := mtl.NewMTLSize(l.contextLength, l.headSize, l.headsCount*batchSize)

	qryObject = device.Reshape(qryObject, reshapeToDims)
	keyObject = device.Reshape(keyObject, reshapeToDims)
	valObject = device.Reshape(valObject, reshapeToDims)

	// Transpose q and v
	qryObject = device.Transpose(qryObject)
	valObject = device.Transpose(valObject)

	// Extract weiObject
	k := float32(math.Pow(float64(l.headSize), -0.5))
	weiObject := device.MatrixMultiply(qryObject, keyObject, k)

	// Apply triangle lower softmax
	weiSoftmax := device.TriangleLowerSoftmax(weiObject)

	// Get MHA-output objects
	bx = device.MatrixMultiply(weiSoftmax, valObject, 1) // bx - horizontal stacked

	// Transpose output before reshape
	bx = device.Transpose(bx) // bx - vertical stacked

	// Reshape output back to big matrix (instead of concatenation)
	bx = device.Reshape(bx, mtl.NewMTLSize(l.contextLength, l.featuresCount, batchSize)) // bx - vertical

	out := device.Transpose(bx) // bx - horizontal

	return out
}

func (l *SAMultiHead) ForUpdate() []*num.Data {
	return l.forUpdate
}

type SAMultiHeadRope2Config struct {
	QryWeights []float32
	KeyWeights []float32
	ValWeights []float32
}

func (l *SAMultiHead) MarshalJSON() ([]byte, error) {
	config := SAMultiHeadRope2Config{
		QryWeights: l.QryWeights.Data.GetFloats(),
		KeyWeights: l.KeyWeights.Data.GetFloats(),
		ValWeights: l.ValWeights.Data.GetFloats(),
	}
	return json.Marshal(config)
}

func (l *SAMultiHead) UnmarshalJSON(bytes []byte) error {
	config := SAMultiHeadRope2Config{
		QryWeights: l.QryWeights.Data.GetFloats(),
		KeyWeights: l.KeyWeights.Data.GetFloats(),
		ValWeights: l.ValWeights.Data.GetFloats(),
	}
	return json.Unmarshal(bytes, &config)
}

func (l *SAMultiHead) LoadFromProvider() {
	l.provideWeights(l.QryWeights, l.KeyWeights, l.ValWeights)
}
