package pkg

import (
	"github.com/atkhx/metal/dataset/cifar-100"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/model"
	"github.com/atkhx/metal/nn/proc"
)

func CreateCIFAR100Model(
	miniBatchSize int,
	device *proc.Device,
	optimizer proc.Optimizer,
) *model.Model {
	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)
	inDims := mtl.MTLSize{
		W: cifar_100.ImageWidth,
		H: cifar_100.ImageHeight,
		D: cifar_100.ImageDepth * miniBatchSize,
	}
	nDims := inDims

	var layers layer.Layers
	{ // Conv block 1
		filterSize, filtersCount, convPadding, convStride := 3, 16, 1, 1
		poolSize, poolStride, poolPadding := 2, 2, 0

		layers = append(layers, layer.Layers{
			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
			layer.NewReLu(),
			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
		}...)

		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
	}

	{ // Conv block 2
		filterSize, filtersCount, convPadding, convStride := 3, 16, 1, 1
		poolSize, poolStride, poolPadding := 2, 2, 0

		layers = append(layers, layer.Layers{
			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
			layer.NewReLu(),
			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
		}...)

		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
	}

	nDims = mtl.NewMTLSize(nDims.Length()/miniBatchSize, 1, miniBatchSize)
	layers = append(layers, layer.NewReshape(nDims))

	//if linearSize > 0 {
	//	layers = append(layers, layer.NewLinear(linearSize, initCfg.Conv, true, nil))
	//	layers = append(layers, layer.NewReLu())
	//}

	layers = append(layers, layer.NewLinear(cifar_100.ClassesCount, initCfg.Linear, false, nil))
	return model.New(inDims, layers, device, optimizer)
}
