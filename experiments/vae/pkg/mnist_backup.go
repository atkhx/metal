package pkg

//func CreateMnistVAEModelBackup(
//	miniBatchSize int,
//	latentDim int,
//	device *proc.Device,
//	optimizer proc.Optimizer,
//) *MnistVAE {
//
//	initCfg := model.DefaultInitConfig(initializer.DistributionUniform)
//
//	inDims := mtl.MTLSize{
//		W: mnist.ImageWidth,
//		H: mnist.ImageHeight,
//		D: mnist.ImageDepth * miniBatchSize,
//	}
//	nDims := inDims
//
//	var layers layer.Layers
//	muLogVarIndex := -1
//
//	// =====================
//	// Encoder
//	// =====================
//
//	{
//		// Conv block 1
//		// In: [28, 28, miniBatchSize]
//		// - conv out: [28, 28, miniBatchSize * filtersCount]
//		// - pool out: [14, 14, miniBatchSize * filtersCount]
//		// Out: [14, 14, miniBatchSize * filtersCount]
//
//		filterSize, filtersCount := 3, 32
//		convPadding, convStride := 1, 1
//		poolSize, poolStride, poolPadding := 2, 2, 0
//
//		layers = append(layers,
//			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
//			layer.NewReLu(),
//			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
//		)
//
//		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
//		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
//	}
//
//	{
//		// Conv block 2
//		// In: [14, 14, miniBatchSize]
//		// - conv out: [14, 14, miniBatchSize * filtersCount]
//		// - pool out: [7, 7, miniBatchSize * filtersCount]
//		// Out: [7, 7, miniBatchSize * filtersCount]
//
//		filterSize, filtersCount := 3, 32
//		convPadding, convStride := 1, 1
//		poolSize, poolStride, poolPadding := 2, 2, 0
//
//		layers = append(layers,
//			layer.NewConv(filterSize, filtersCount, miniBatchSize, convPadding, convStride, initCfg.Conv, nil),
//			layer.NewReLu(),
//			layer.NewMaxPool2D(poolSize, poolStride, poolPadding),
//		)
//
//		nDims = device.GetConvSize(nDims.W, filterSize, filtersCount, miniBatchSize, convPadding, convStride)
//		nDims = device.GetPoolSize(nDims, poolSize, poolPadding, poolStride)
//	}
//
//	// Flatten
//	flatSize := nDims.Length() / miniBatchSize
//	nDims = mtl.NewMTLSize(flatSize, 1, miniBatchSize)
//	layers = append(layers, layer.NewReshape(nDims))
//
//	// μ and logσ² (concat)
//	layers = append(layers,
//		layer.NewLinear(latentDim*2, initCfg.Linear, true, nil), // [μ | logσ²]
//	)
//	muLogVarIndex = len(layers) - 1
//
//	// =====================
//	// Sampling layer
//	// =====================
//	layers = append(layers,
//		layer.NewVAESample(latentDim),
//	)
//
//	// =====================
//	// Decoder
//	// =====================
//
//	// Linear → reshape обратно в feature map
//	layers = append(layers,
//		layer.NewLinear(flatSize, initCfg.Linear, true, nil),
//	)
//
//	nDims = mtl.NewMTLSize(nDims.W, nDims.H, miniBatchSize)
//	layers = append(layers,
//		layer.NewReshape(mtl.MTLSize{
//			W: 7,
//			H: 7,
//			D: 32 * miniBatchSize,
//		}),
//	)
//
//	// Upsample + Conv
//	layers = append(layers,
//		layer.NewUpSample2D(2), // 7 → 14
//		layer.NewConv(3, 32, miniBatchSize, 1, 1, initCfg.Conv, nil),
//		layer.NewReLu(),
//
//		layer.NewUpSample2D(2), // 14 → 28
//		layer.NewConv(3, 1, miniBatchSize, 1, 1, initCfg.Conv, nil),
//		layer.NewSigmoid(), // MNIST
//	)
//
//	return &MnistVAE{
//		Model:         model.New(inDims, layers, device, optimizer),
//		muLogVarIndex: muLogVarIndex,
//	}
//}
