package vaephoto

import (
	"fmt"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/model"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

const (
	PhotoPatchSize      = 100
	PhotoLatentDim      = 256
	PhotoBatchSize      = 10
	PhotoConvFiltersPre = 64
	PhotoConvFiltersMid = 64
)

type PhotoVAE struct {
	*model.Model
	muLogVar  *layer.Linear
	vaeSample *layer.VAESample
}

func (m *PhotoVAE) GetMuLogVar() *num.Data {
	return m.muLogVar.GetOutput()
}

func (m *PhotoVAE) GetVAESample() *num.Data {
	return m.vaeSample.GetOutput()
}

func CreatePhotoVAETrainModel(
	miniBatchSize int,
	latentDim int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *PhotoVAE {
	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)

	inDims := mtl.MTLSize{
		W: PhotoPatchSize,
		H: PhotoPatchSize,
		D: ImageDepthRGB * miniBatchSize,
	}

	var layers layer.Layers

	encoder, flatSize, _ := createPhotoEncoderBlock(inDims, initCfg, miniBatchSize, device)
	layers = append(layers, encoder)

	muLogVar := layer.NewLinear(latentDim*2, initCfg.Linear, true, nil)
	vaeSample := layer.NewVAESample(latentDim)

	layers = append(layers,
		muLogVar,
		vaeSample,
		createPhotoDecoderBlock(initCfg, miniBatchSize, flatSize),
	)

	return &PhotoVAE{
		Model:     model.New(inDims, layers, device, optimizer),
		muLogVar:  muLogVar,
		vaeSample: vaeSample,
	}
}

func CreatePhotoVAEInferenceModel(
	miniBatchSize int,
	latentDim int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *PhotoVAE {
	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)

	inDims := mtl.MTLSize{W: PhotoPatchSize, H: PhotoPatchSize, D: ImageDepthRGB * miniBatchSize}
	_, flatSize, _ := createPhotoEncoderBlock(inDims, initCfg, miniBatchSize, device)

	nDims := mtl.NewMTLSize(latentDim, 1, miniBatchSize)

	return &PhotoVAE{
		Model: model.New(nDims, layer.Layers{
			&layer.NoopLayer{},
			&layer.NoopLayer{},
			&layer.NoopLayer{},
			createPhotoDecoderBlock(initCfg, miniBatchSize, flatSize),
		}, device, optimizer),
	}
}

func createPhotoDecoderBlock(
	initCfg model.InitConfig,
	miniBatchSize int,
	flatSize int,
) *layer.VAEDecoder {
	return &layer.VAEDecoder{Layers: layer.Layers{
		layer.NewLinear(flatSize, initCfg.Linear, true, nil),
		layer.NewReshape(mtl.MTLSize{
			W: PhotoPatchSize / 4, // 24 // 25,
			H: PhotoPatchSize / 4, // 24 // 25,
			D: PhotoConvFiltersPre * miniBatchSize},
		),

		layer.NewUpSample2D(2), // 12 -> 24 // 25 → 50
		layer.NewConv(3, PhotoConvFiltersMid, miniBatchSize, 1, 1, initCfg.Conv, nil),
		layer.NewReLu(),

		layer.NewUpSample2D(2), // 50 → 100
		layer.NewConv(3, ImageDepthRGB, miniBatchSize, 1, 1, initCfg.Conv, nil),
		layer.NewSigmoid(),
	}}
}

func createPhotoEncoderBlock(
	nDims mtl.MTLSize,
	initCfg model.InitConfig,
	miniBatchSize int,
	device *proc.Device,
) (*layer.VAEEncoder, int, mtl.MTLSize) {
	var layers layer.Layers
	{ // Conv block 1: [100, 100, batch*3] -> [50, 50, batch*filters]
		filterSize, filtersCount := 3, PhotoConvFiltersPre
		convPadding, convStride := 1, 1
		poolSize, poolStride, poolPadding := 2, 2, 0
		layers = append(layers,
			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
			layer.NewReLu(),
			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
		)
		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
		fmt.Println("nDims", nDims)
		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
		fmt.Println("nDims", nDims)
	}
	{ // Conv block 2: [50, 50, batch*filters] -> [25, 25, batch*filters]
		filterSize, filtersCount := 3, PhotoConvFiltersPre
		convPadding, convStride := 1, 1
		poolSize, poolStride, poolPadding := 2, 2, 0

		layers = append(layers,
			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
			layer.NewReLu(),
			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
		)

		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
		fmt.Println("nDims", nDims)
		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
		fmt.Println("nDims", nDims)
	}

	fmt.Println("------------------------------")
	flatSize := nDims.Length() / miniBatchSize
	nDims = mtl.NewMTLSize(flatSize, 1, miniBatchSize)
	layers = append(layers, layer.NewReshape(nDims))
	return &layer.VAEEncoder{Layers: layers}, flatSize, nDims
}
