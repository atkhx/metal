package model

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/proc"
)

type ConvLayerCfg struct {
	Size    int
	Count   int
	Padding int
	Stride  int
}

func NewConv(
	convLayersCfg []ConvLayerCfg,
	imageSize int,
	imageDepth int,
	miniBatchSize int,
	classesCount int,
	filtersCount int,
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

	cfg := normalizeInitConfig(initCfg)

	convInitializer := cfg.Conv

	var layers layer.Layers

	// Create conv layers
	for _, convLayerCfg := range convLayersCfg {
		layers = append(layers, layer.NewConv(
			convLayerCfg.Size,
			convLayerCfg.Count,
			miniBatchSize,
			convLayerCfg.Padding,
			convLayerCfg.Stride,
			convInitializer,
			nil,
		))
		// TODO Add pooling layer
		// Add ReLU activation layer
		layers = append(layers, layer.NewReLu())

		// Store convLayer output dimensions
		oDims = device.GetConvSize(
			oDims.W,
			convLayerCfg.Size,
			convLayerCfg.Count,
			miniBatchSize,
			convLayerCfg.Padding,
			convLayerCfg.Stride,
		)
	}

	// Reshape to rows x miniBatchSize
	layers = append(layers, layer.NewReshape(
		mtl.NewMTLSize(oDims.W*oDims.W*filtersCount, 1, miniBatchSize),
	))

	// Add linear layer
	if linearSize > 0 {
		layers = append(layers, layer.NewLinear(linearSize, cfg.Conv, true, nil))
		layers = append(layers, layer.NewReLu())
	}

	// Prepare for classification classesCount x miniBatchSize
	layers = append(layers, layer.NewLinear(classesCount, cfg.Linear, false, nil))
	return New(inDims, layers, device, modelOptimizer)
}
