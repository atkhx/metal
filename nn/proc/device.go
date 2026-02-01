package proc

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/ops/adamw"
	"github.com/atkhx/metal/nn/ops/addcols"
	"github.com/atkhx/metal/nn/ops/addequal"
	"github.com/atkhx/metal/nn/ops/addrows"
	"github.com/atkhx/metal/nn/ops/conv"
	"github.com/atkhx/metal/nn/ops/dropout"
	"github.com/atkhx/metal/nn/ops/embeddings"
	"github.com/atkhx/metal/nn/ops/gelu"
	"github.com/atkhx/metal/nn/ops/gelunew"
	"github.com/atkhx/metal/nn/ops/layernormrows"
	"github.com/atkhx/metal/nn/ops/matmul"
	"github.com/atkhx/metal/nn/ops/maxpool"
	"github.com/atkhx/metal/nn/ops/mean"
	"github.com/atkhx/metal/nn/ops/mulcols"
	"github.com/atkhx/metal/nn/ops/mulequal"
	"github.com/atkhx/metal/nn/ops/mulrows"
	"github.com/atkhx/metal/nn/ops/nllpos"
	"github.com/atkhx/metal/nn/ops/positionaladd"
	"github.com/atkhx/metal/nn/ops/relu"
	"github.com/atkhx/metal/nn/ops/rmsnormrows"
	"github.com/atkhx/metal/nn/ops/ropecols"
	"github.com/atkhx/metal/nn/ops/sanitize"
	"github.com/atkhx/metal/nn/ops/silu"
	"github.com/atkhx/metal/nn/ops/softmax"
	"github.com/atkhx/metal/nn/ops/transpose"
	"github.com/atkhx/metal/nn/ops/trilmask"
	"github.com/atkhx/metal/nn/pipeline"
)

type Device struct {
	mtlDevice *mtl.Device
}

func New(mtlDevice *mtl.Device) *Device {
	return &Device{mtlDevice: mtlDevice}
}

func NewWithSystemDefaultDevice() *Device {
	return &Device{mtl.MustCreateSystemDefaultDevice()}
}

func (d *Device) Release() {
	d.mtlDevice.Release()
}

func (d *Device) GetMTLDevice() *mtl.Device {
	return d.mtlDevice
}

func (d *Device) GetTestingPipeline(lastNode *num.Data) *pipeline.TestingPipeline {
	return pipeline.NewTestingPipeline(d.mtlDevice, lastNode)
}

func (d *Device) GetInferencePipeline(lastNode *num.Data) *pipeline.InferencePipeline {
	return pipeline.NewInferencePipeline(d.mtlDevice, lastNode)
}

func (d *Device) GetTrainingPipeline(lastNode *num.Data) *pipeline.TrainingPipeline {
	return pipeline.NewTrainingPipeline(d.mtlDevice, lastNode)
}

func (d *Device) NewData(dims mtl.MTLSize, deps ...*num.Data) *num.Data {
	return &num.Data{
		Data: d.mtlDevice.NewBufferEmptyFloatsBuffer(dims.Length(), mtl.ResourceStorageModeShared),
		Grad: d.mtlDevice.NewBufferEmptyFloatsBuffer(dims.Length(), mtl.ResourceStorageModeShared),
		Dims: dims,
		Deps: deps,
	}
}

func (d *Device) NewDataWithValues(dims mtl.MTLSize, values []float32) *num.Data {
	return &num.Data{
		Data: d.mtlDevice.NewBufferWithFloats(values, mtl.ResourceStorageModeShared),
		Grad: d.mtlDevice.NewBufferEmptyFloatsBuffer(dims.Length(), mtl.ResourceStorageModeShared),
		Dims: dims,
	}
}

func (d *Device) newLinkedCopy(data *num.Data, links ...*num.Data) *num.Data {
	return d.NewData(data.Dims, append([]*num.Data{data}, links...)...)
}

// NewDataRandUniformWeighted fills data with U(-w, w).
func (d *Device) NewDataRandUniformWeighted(dims mtl.MTLSize, w float32) *num.Data {
	data := make([]float32, dims.Length())
	for i := range data {
		data[i] = float32(rand.Float64())*2*w - w
	}
	return d.NewDataWithValues(dims, data)
}

// NewDataRandNormalWeighted fills data with N(0, w^2).
func (d *Device) NewDataRandNormalWeighted(dims mtl.MTLSize, w float32) *num.Data {
	data := make([]float32, dims.Length())
	for i := range data {
		data[i] = float32(rand.NormFloat64()) * w
	}
	return d.NewDataWithValues(dims, data)
}

func (d *Device) NewTokenEmbeddingTable(featuresCount, alphabetSize int, w float32) *num.Data {
	return d.NewDataRandUniformWeighted(mtl.NewMTLSize(featuresCount, alphabetSize), w)
}

type Kernel interface {
	Forward(b *mtl.CommandBuffer)
	Backward(b *mtl.CommandBuffer)
}

func (d *Device) assocKernel(output *num.Data, kernel Kernel) *num.Data {
	output.CalcData = kernel.Forward
	output.CalcGrad = kernel.Backward
	return output
}

func (d *Device) Mean(input *num.Data) *num.Data {
	output := d.NewData(mtl.NewMTLSize(), input)
	kernel := mean.New(d.mtlDevice, input, output, input.Dims.Length())
	return d.assocKernel(output, kernel)
}

func (d *Device) Add(aData, bData *num.Data) *num.Data {
	if aData.Dims == bData.Dims {
		return d.AddEqual(aData, bData)
	}
	if aData.Dims.W == bData.Dims.W && bData.Dims.H == 1 && bData.Dims.D == 1 {
		return d.AddRow(aData, bData, aData.Dims.W)
	}
	panic("not implemented")
}

func (d *Device) AddEqual(input, weights *num.Data) *num.Data {
	if input.Dims != weights.Dims {
		panic("dimensions are not equal")
	}
	output := d.newLinkedCopy(input, weights)
	kernel := addequal.New(d.mtlDevice, input, weights, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) AddRow(input, weights *num.Data, width int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	kernel := addrows.New(d.mtlDevice, input, weights, output, width)
	return d.assocKernel(output, kernel)
}

func (d *Device) AddCol(input, weights *num.Data, width, height int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	kernel := addcols.New(d.mtlDevice, input, weights, output, width, height)
	return d.assocKernel(output, kernel)
}

func (d *Device) MulCol(input, weights *num.Data, width, height int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	kernel := mulcols.New(d.mtlDevice, input, weights, output, width, height)
	return d.assocKernel(output, kernel)
}

func (d *Device) MulRow(input, weights *num.Data, width int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	kernel := mulrows.New(d.mtlDevice, input, weights, output, width)
	return d.assocKernel(output, kernel)
}

func (d *Device) MulEqual(input, weights *num.Data) *num.Data {
	if input.Dims.Length() != weights.Dims.Length() {
		panic("input size != weights size")
	}
	output := d.newLinkedCopy(input, weights)
	kernel := mulequal.New(d.mtlDevice, input, weights, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) RMSNorm(input *num.Data, width int) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := rmsnormrows.New(d.mtlDevice, input, output, width)
	return d.assocKernel(output, kernel)
}

func (d *Device) LayerNorm(input *num.Data, width int, eps float32) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := layernormrows.New(d.mtlDevice, input, output, width, eps)
	return d.assocKernel(output, kernel)
}

func (d *Device) RopeCols(input *num.Data, featuresCount, headSize, contextLength int) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := ropecols.New(d.mtlDevice, input, output, featuresCount, headSize, contextLength)
	return d.assocKernel(output, kernel)
}

func (d *Device) PositionalAdd(input, weights *num.Data, colsCount, rowsCount int) *num.Data {
	output := d.newLinkedCopy(input, weights)
	kernel := positionaladd.New(d.mtlDevice, input, weights, output, colsCount, rowsCount)
	return d.assocKernel(output, kernel)
}

func (d *Device) Relu(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := relu.New(d.mtlDevice, input, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) Sanitize(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := sanitize.New(d.mtlDevice, input, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) SiLu(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := silu.New(d.mtlDevice, input, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) GeLu(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := gelu.New(d.mtlDevice, input, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) GeLuNew(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := gelunew.New(d.mtlDevice, input, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) Dropout(input *num.Data, prob float32) *num.Data {
	if prob == 0 {
		return input
	}

	output := d.newLinkedCopy(input)
	kernel := dropout.New(d.mtlDevice, input, output, prob, uint64(time.Now().UnixNano()))
	return d.assocKernel(output, kernel)
}

func (d *Device) Reshape(input *num.Data, dims mtl.MTLSize) *num.Data {
	if input.Dims.Length() != dims.Length() {
		panic("total dimension size must be equal with original")
	}

	output := *input
	output.Dims = dims
	output.Deps = []*num.Data{input}
	output.SkipResetGrad = true
	output.CalcData = func(b *mtl.CommandBuffer) {}
	output.CalcGrad = func(b *mtl.CommandBuffer) {}
	return &output
}

func (d *Device) TrilMask(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := trilmask.New(d.mtlDevice, input, output, input.Dims.W, input.Dims.H)
	return d.assocKernel(output, kernel)
}

func (d *Device) Softmax(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	kernel := softmax.New(d.mtlDevice, input, output)
	return d.assocKernel(output, kernel)
}

func (d *Device) TriangleLowerSoftmax(input *num.Data) *num.Data {
	return d.Softmax(d.TrilMask(input))
}

func (d *Device) Transpose(input *num.Data) *num.Data {
	output := d.newLinkedCopy(input)
	output.Dims.W = input.Dims.H
	output.Dims.H = input.Dims.W

	kernel := transpose.New(d.mtlDevice, input, output, input.Dims.W, input.Dims.H)
	return d.assocKernel(output, kernel)
}

func (d *Device) Embeddings(input *num.Data, tEmbeddings *num.Data) *num.Data {
	featuresCount := tEmbeddings.Dims.W
	contextSize := input.Dims.W
	batchSize := input.Dims.H

	fmt.Println("Embeddings input.Dims", input.Dims)
	fmt.Println("Embeddings tEmbeddings.Dims", tEmbeddings.Dims)
	fmt.Println("featuresCount", featuresCount)
	fmt.Println("contextSize", contextSize)
	fmt.Println()

	output := d.NewData(mtl.NewMTLSize(featuresCount, contextSize, batchSize), tEmbeddings)
	kernel := embeddings.New(d.mtlDevice, input, output, tEmbeddings, featuresCount, contextSize)
	return d.assocKernel(output, kernel)
}

func (d *Device) CrossEntropyPos(input, targets *num.Data) *num.Data {
	if targets.Dims.W != 1 {
		panic("target width must be equal 1")
	}

	if targets.Dims.H != input.Dims.H {
		fmt.Println("targets.Dims.H", targets.Dims.H)
		fmt.Println("input.Dims.H", input.Dims.H)
		panic("target height must be equal input height")
	}

	if targets.Dims.D != input.Dims.D {
		fmt.Println("targets.Dims.D", targets.Dims.D)
		fmt.Println("input.Dims.D", input.Dims.D)
		panic("target depth must be equal input depth")
	}

	inputSoftmax := d.Softmax(d.Sanitize(input))
	output := d.NewData(targets.Dims, inputSoftmax)
	kernel := nllpos.New(d.mtlDevice, inputSoftmax, output, targets, input.Dims.W)
	return d.assocKernel(output, kernel)
}

func (d *Device) MatrixMultiply(aData, bData *num.Data, alpha float32) *num.Data {
	if aData.Dims.W != bData.Dims.H {
		panic("aData width must be equal bData height")
	}

	if aData.Dims.D != bData.Dims.D && !(aData.Dims.D == 1 || bData.Dims.D == 1) {
		panic("aData's and bData's dept must be equal or one of them must be 1")
	}

	oD := aData.Dims.D
	if bData.Dims.D > oD {
		oD = bData.Dims.D
	}

	oW := bData.Dims.W
	oH := aData.Dims.H

	output := d.NewData(mtl.MTLSize{W: oW, H: oH, D: oD}, aData, bData)
	kernel := matmul.New(d.mtlDevice, aData, bData, output, alpha)
	return d.assocKernel(output, kernel)
}

func (d *Device) GetConvSize(imageSize, filterSize, filtersCount, batchSize, padding, stride int) mtl.MTLSize {
	ow := (imageSize-filterSize+2*padding)/stride + 1
	oh := (imageSize-filterSize+2*padding)/stride + 1
	od := filtersCount * batchSize

	return mtl.MTLSize{W: ow, H: oh, D: od}
}

func (d *Device) GetPoolSize(width, height, poolSize, padding, stride int) mtl.MTLSize {
	ow := (width-poolSize+2*padding)/stride + 1
	oh := (height-poolSize+2*padding)/stride + 1
	return mtl.MTLSize{W: ow, H: oh, D: 1}
}

func (d *Device) Conv(input, weights, biases *num.Data, filtersCount, batchSize, padding, stride int) *num.Data {
	convSize := d.GetConvSize(input.Dims.W, weights.Dims.W, filtersCount, batchSize, padding, stride)
	output := d.NewData(convSize, input, weights, biases)
	kernel := conv.New(d.mtlDevice, input, weights, biases, output, filtersCount, batchSize, padding, stride)
	return d.assocKernel(output, kernel)
}

func (d *Device) MaxPool2D(input *num.Data, poolSize, padding, stride int) *num.Data {
	if poolSize < 1 {
		panic("poolSize must be >= 1")
	}
	if stride < 1 {
		panic("stride must be >= 1")
	}
	if padding < 0 {
		panic("padding must be >= 0")
	}

	out := d.GetPoolSize(input.Dims.W, input.Dims.H, poolSize, padding, stride)
	out.D = input.Dims.D

	output := d.NewData(out, input)
	kernel := maxpool.New(d.mtlDevice, input, output, poolSize, stride, padding)
	return d.assocKernel(output, kernel)
}

type Optimize func(b *mtl.CommandBuffer, iteration int)
type Optimizer func(nodes []*num.Data) Optimize

func (d *Device) GetOptimizerAdam(iterations int, beta1, beta2, learningRate, eps float32) Optimizer {
	iterations++
	kernel := adamw.New(d.mtlDevice)

	return func(nodes []*num.Data) Optimize {

		mm := make([]*mtl.Buffer, len(nodes))
		vv := make([]*mtl.Buffer, len(nodes))

		for i, node := range nodes {
			mm[i] = d.mtlDevice.NewBufferEmptyFloatsBuffer(node.Dims.Length(), mtl.ResourceStorageModeShared)
			vv[i] = d.mtlDevice.NewBufferEmptyFloatsBuffer(node.Dims.Length(), mtl.ResourceStorageModeShared)
		}

		beta1pow := make([]float32, iterations)
		beta2pow := make([]float32, iterations)

		for i := 0; i < iterations; i++ {
			if i == 0 {
				beta1pow[i] = beta1
				beta2pow[i] = beta2
			} else {
				beta1pow[i] = beta1pow[i-1] * beta1
				beta2pow[i] = beta2pow[i-1] * beta2
			}
		}

		for i := 0; i < iterations; i++ {
			beta1pow[i] = 1 / (1 - beta1pow[i])
			beta2pow[i] = 1 / (1 - beta2pow[i])
		}

		return func(b *mtl.CommandBuffer, iteration int) {
			beta1powIterationLR := learningRate * beta1pow[iteration]
			beta2powIteration := beta2pow[iteration]

			for i, node := range nodes {
				kernel.UpdateWithAdam(
					b,
					node.Data,
					node.Grad,
					mm[i],
					vv[i],
					beta1,
					beta2,
					beta1powIterationLR,
					beta2powIteration,
					eps,
				)
			}
		}
	}
}
