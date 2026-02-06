package pkg

import (
	"github.com/atkhx/metal/dataset/cifar-10"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/model"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

const (
	CIFARLatentDim       = 128
	CIFARBatchSize       = 16
	CIFARConvFiltersPre  = 128
	CIFARConvFiltersPost = 64
)

type CifarVAE struct {
	*model.Model
	muLogVar  *layer.Linear
	vaeSample *layer.VAESample
}

func (m *CifarVAE) GetMuLogVar() *num.Data {
	return m.muLogVar.GetOutput()
}

func (m *CifarVAE) GetVAESample() *num.Data {
	return m.vaeSample.GetOutput()
}

func CreateCifarVAETrainModel(
	miniBatchSize int,
	latentDim int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *CifarVAE {
	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)

	inDims := mtl.MTLSize{
		W: cifar_10.ImageWidth,
		H: cifar_10.ImageHeight,
		D: cifar_10.ImageDepthRGB * miniBatchSize,
	}

	var layers layer.Layers

	encoder, flatSize, _ := createCifarEncoderBlock(inDims, initCfg, miniBatchSize, device)
	layers = append(layers, encoder)

	muLogVar := layer.NewLinear(latentDim*2, initCfg.Linear, true, nil)
	vaeSample := layer.NewVAESample(latentDim)

	layers = append(layers,
		muLogVar,
		vaeSample,
		createCifarDecoderBlock(initCfg, miniBatchSize, flatSize),
	)

	return &CifarVAE{
		Model:     model.New(inDims, layers, device, optimizer),
		muLogVar:  muLogVar,
		vaeSample: vaeSample,
	}
}

func CreateCifarVAEInferenceModel(
	miniBatchSize int,
	latentDim int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *CifarVAE {
	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)

	inDims := mtl.MTLSize{W: cifar_10.ImageWidth, H: cifar_10.ImageHeight, D: cifar_10.ImageDepthRGB * miniBatchSize}
	_, flatSize, _ := createCifarEncoderBlock(inDims, initCfg, miniBatchSize, device)

	nDims := mtl.NewMTLSize(latentDim, 1, miniBatchSize)

	return &CifarVAE{
		Model: model.New(nDims, layer.Layers{
			&layer.NoopLayer{},
			&layer.NoopLayer{},
			&layer.NoopLayer{},
			createCifarDecoderBlock(initCfg, miniBatchSize, flatSize),
		}, device, optimizer),
	}
}

func createCifarDecoderBlock(
	initCfg model.InitConfig,
	miniBatchSize int,
	flatSize int,
) *layer.VAEDecoder {
	return &layer.VAEDecoder{Layers: layer.Layers{
		layer.NewLinear(flatSize, initCfg.Linear, true, nil),
		layer.NewReshape(mtl.MTLSize{W: 8, H: 8, D: CIFARConvFiltersPre * miniBatchSize}),

		layer.NewUpSample2D(2), // 8 → 16
		layer.NewConv(3, CIFARConvFiltersPost, miniBatchSize, 1, 1, initCfg.Conv, nil),
		layer.NewReLu(),

		layer.NewUpSample2D(2), // 16 → 32
		layer.NewConv(3, cifar_10.ImageDepthRGB, miniBatchSize, 1, 1, initCfg.Conv, nil),
		layer.NewSigmoid(),
	}}
}

func createCifarEncoderBlock(
	nDims mtl.MTLSize,
	initCfg model.InitConfig,
	miniBatchSize int,
	device *proc.Device,
) (*layer.VAEEncoder, int, mtl.MTLSize) {
	var layers layer.Layers
	{ // Conv block 1: [32, 32, batch*3] -> [16, 16, batch*filters]
		filterSize, filtersCount := 3, CIFARConvFiltersPre
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
	{ // Conv block 2: [16, 16, batch*filters] -> [8, 8, batch*filters]
		filterSize, filtersCount := 3, CIFARConvFiltersPre
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

	flatSize := nDims.Length() / miniBatchSize
	nDims = mtl.NewMTLSize(flatSize, 1, miniBatchSize)
	layers = append(layers, layer.NewReshape(nDims))
	return &layer.VAEEncoder{Layers: layers}, flatSize, nDims
}
