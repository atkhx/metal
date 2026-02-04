package pkg

import (
	"github.com/atkhx/metal/dataset/mnist"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/model"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

const (
	MNISTBatchSize = 10

	MNISTLatentDim       = 32
	MNISTConvFiltersPre  = 64
	MNISTConvFiltersPost = 32
)

type MnistVAE struct {
	*model.Model
	muLogVar  *layer.Linear
	vaeSample *layer.VAESample
}

func (m *MnistVAE) GetMuLogVar() *num.Data {
	return m.muLogVar.GetOutput()
}

func (m *MnistVAE) GetVAESample() *num.Data {
	return m.vaeSample.GetOutput()
}

func CreateMnistVAETrainModel(
	miniBatchSize int,
	latentDim int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *MnistVAE {
	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)

	inDims := mtl.MTLSize{
		W: mnist.ImageWidth,
		H: mnist.ImageHeight,
		D: mnist.ImageDepth * miniBatchSize,
	}

	var layers layer.Layers

	encoder, flatSize, _ := createEncoderBlock(inDims, initCfg, miniBatchSize, device)
	layers = append(layers, encoder)

	muLogVar := layer.NewLinear(latentDim*2, initCfg.Linear, true, nil)
	vaeSample := layer.NewVAESample(latentDim)

	layers = append(layers,
		muLogVar,
		vaeSample,
		createDecoderBlock(initCfg, miniBatchSize, flatSize),
	)

	return &MnistVAE{
		Model:     model.New(inDims, layers, device, optimizer),
		muLogVar:  muLogVar,
		vaeSample: vaeSample,
	}
}

func CreateMnistVAEInferenceModel(
	miniBatchSize int,
	latentDim int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *MnistVAE {
	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)

	inDims := mtl.MTLSize{W: mnist.ImageWidth, H: mnist.ImageHeight, D: mnist.ImageDepth * miniBatchSize}
	_, flatSize, nDims := createEncoderBlock(inDims, initCfg, miniBatchSize, device)

	//muLogVar := layer.NewLinear(latentDim*2, initCfg.Linear, true, nil)

	nDims = mtl.NewMTLSize(latentDim, 1, miniBatchSize)

	return &MnistVAE{
		Model: model.New(nDims, layer.Layers{
			&layer.NoopLayer{},
			&layer.NoopLayer{}, //muLogVar,
			&layer.NoopLayer{}, // layer.NewVAESample(latentDim),
			createDecoderBlock(initCfg, miniBatchSize, flatSize),
		}, device, optimizer),
		//muLogVar: muLogVar,
	}
}

func createDecoderBlock(
	initCfg model.InitConfig,
	miniBatchSize int,
	flatSize int,
) *layer.VAEDecoder {
	return &layer.VAEDecoder{Layers: layer.Layers{
		layer.NewLinear(flatSize, initCfg.Linear, true, nil),
		layer.NewReshape(mtl.MTLSize{W: 7, H: 7, D: MNISTConvFiltersPre * miniBatchSize}),

		layer.NewUpSample2D(2), // 7 → 14
		layer.NewConv(3, MNISTConvFiltersPost, miniBatchSize, 1, 1, initCfg.Conv, nil),
		layer.NewReLu(),

		layer.NewUpSample2D(2), // 14 → 28
		layer.NewConv(3, 1, miniBatchSize, 1, 1, initCfg.Conv, nil),
		layer.NewSigmoid(), // MNIST
	}}
}

func createEncoderBlock(
	nDims mtl.MTLSize,
	initCfg model.InitConfig,
	miniBatchSize int,
	device *proc.Device,
) (*layer.VAEEncoder, int, mtl.MTLSize) {
	var layers layer.Layers
	{ // Conv block 1: [28, 28, miniBatchSize] -> [14, 14, miniBatchSize * filtersCount]
		filterSize, filtersCount := 3, MNISTConvFiltersPre
		convPadding, convStride := 1, 1
		poolSize, poolStride, poolPadding := 2, 2, 0
		layers = append(layers,
			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
			layer.NewReLu(),
			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
		)
		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
	}
	{ // Conv block 2: [14, 14, miniBatchSize] -> [7, 7, miniBatchSize * filtersCount]
		filterSize, filtersCount := 3, MNISTConvFiltersPre
		convPadding, convStride := 1, 1
		poolSize, poolStride, poolPadding := 2, 2, 0

		layers = append(layers,
			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
			layer.NewReLu(),
			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
		)

		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
	}
	// Flatten
	// [7, 7, miniBatchSize * filtersCount]
	flatSize := nDims.Length() / miniBatchSize
	nDims = mtl.NewMTLSize(flatSize, 1, miniBatchSize)
	// [7 x 7 x filtersCount, 1, miniBatchSize]
	// [7 x 7 x 64, 1, miniBatchSize]
	layers = append(layers, layer.NewReshape(nDims))
	return &layer.VAEEncoder{Layers: layers}, flatSize, nDims
}
