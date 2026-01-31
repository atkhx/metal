package model

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/proc"
)

func NewCNN(
	imageSize int,
	imageDepth int,
	miniBatchSize int,
	classesCount int,
	filterSize int,
	filtersCount int,
	padding int,
	convLayersCount int,
	linearSize int,
	initCfg *InitConfig,
	device *proc.Device,
	modelOptimizer proc.Optimizer,
) *Model {
	inDims := mtl.MTLSize{
		W: imageSize,
		H: imageSize,
		D: imageDepth * miniBatchSize,
	}
	oDims := inDims
	stride := 1

	cfg := normalizeInitConfig(initCfg)

	var layers layer.Layers
	for i := 0; i < convLayersCount; i++ {
		layers = append(layers, layer.NewConv(
			filterSize,
			filtersCount,
			miniBatchSize,
			padding,
			stride,
			cfg.Conv,
			nil,
		))
		layers = append(layers, layer.NewReLu())
		oDims = device.GetConvSize(oDims.W, filterSize, filtersCount, padding, stride)
	}

	nDims := mtl.NewMTLSize(oDims.W*oDims.W*filtersCount, 1, miniBatchSize)
	layers = append(layers, layer.NewReshape(nDims))

	if linearSize > 0 {
		layers = append(layers, layer.NewLinear(linearSize, cfg.Conv, true, nil))
		layers = append(layers, layer.NewReLu())
	}

	layers = append(layers, layer.NewLinear(classesCount, cfg.Linear, false, nil))
	return New(inDims, layers, device, modelOptimizer)
}
