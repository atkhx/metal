package layer

import (
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
	initWeights initializer.Initializer,
	provideWeights func(qw, kw, vw, qb, kb, vb *num.Data),
) *SAMultiHeadWithBias {
	if initWeights == nil {
		initWeights = initializer.XavierNormalLinear
	}
	return &SAMultiHeadWithBias{
		initWeights:    initWeights,
		featuresCount:  featuresCount,
		contextLength:  contextLength,
		headsCount:     headsCount,
		headSize:       headSize,
		provideWeights: provideWeights,
	}
}

type SAMultiHeadWithBias struct {
	QryWeights *num.Data
	KeyWeights *num.Data
	ValWeights *num.Data
	QryBias    *num.Data
	KeyBias    *num.Data
	ValBias    *num.Data

	forUpdate []*num.Data

	initWeights    initializer.Initializer
	provideWeights func(qw, kw, vw, qb, kb, vb *num.Data)

	featuresCount int
	contextLength int
	headsCount    int
	headSize      int
}

func (l *SAMultiHeadWithBias) Compile(device *proc.Device, input *num.Data) *num.Data {
	fanIn := input.Dims.Length()
	fanOut := l.featuresCount
	batchSize := input.Dims.D

	l.QryWeights = initWeights(device, l.initWeights, mtl.NewMTLSize(l.featuresCount, l.featuresCount), fanIn, fanOut)
	l.KeyWeights = initWeights(device, l.initWeights, mtl.NewMTLSize(l.featuresCount, l.featuresCount), fanIn, fanOut)
	l.ValWeights = initWeights(device, l.initWeights, mtl.NewMTLSize(l.featuresCount, l.featuresCount), fanIn, fanOut)
	l.QryBias = device.NewData(mtl.NewMTLSize(l.featuresCount))
	l.KeyBias = device.NewData(mtl.NewMTLSize(l.featuresCount))
	l.ValBias = device.NewData(mtl.NewMTLSize(l.featuresCount))

	l.forUpdate = []*num.Data{l.QryWeights, l.KeyWeights, l.ValWeights, l.QryBias, l.KeyBias, l.ValBias}

	bx := device.Transpose(input) // bx - vertical

	// Extract qkv-objects
	qryObject := device.MatrixMultiply(l.QryWeights, bx, 1)
	keyObject := device.MatrixMultiply(l.KeyWeights, bx, 1)
	valObject := device.MatrixMultiply(l.ValWeights, bx, 1)

	// Add biases by rows (features)
	qryObject = device.AddCol(qryObject, l.QryBias, qryObject.Dims.W, qryObject.Dims.H)
	keyObject = device.AddCol(keyObject, l.KeyBias, keyObject.Dims.W, keyObject.Dims.H)
	valObject = device.AddCol(valObject, l.ValBias, valObject.Dims.W, valObject.Dims.H)

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

func (l *SAMultiHeadWithBias) ForUpdate() []*num.Data {
	return l.forUpdate
}

func (l *SAMultiHeadWithBias) LoadFromProvider() {
	l.provideWeights(l.QryWeights, l.KeyWeights, l.ValWeights, l.QryBias, l.KeyBias, l.ValBias)
}
