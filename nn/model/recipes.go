package model

import (
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/num"
)

type InitConfig struct {
	Distribution initializer.Distribution
	Conv         initializer.Initializer
	Linear       initializer.Initializer
	Attention    initializer.Initializer
	FFN          initializer.Initializer
}

func DefaultInitConfig(dist initializer.Distribution) InitConfig {
	switch dist {
	case initializer.DistributionUniform:
		return InitConfig{
			Distribution: dist,
			Conv:         initializer.KaimingUniformReLU,
			Linear:       initializer.XavierUniformLinear,
			Attention:    initializer.XavierUniformLinear,
			FFN:          initializer.KaimingUniformReLU,
		}
	default:
		return InitConfig{
			Distribution: initializer.DistributionNormal,
			Conv:         initializer.KaimingNormalReLU,
			Linear:       initializer.XavierNormalLinear,
			Attention:    initializer.XavierNormalLinear,
			FFN:          initializer.KaimingNormalReLU,
		}
	}
}

func DefaultInitConfigNormal() InitConfig {
	return DefaultInitConfig(initializer.DistributionNormal)
}

func DefaultInitConfigUniform() InitConfig {
	return DefaultInitConfig(initializer.DistributionUniform)
}

func normalizeInitConfig(cfg *InitConfig) InitConfig {
	base := DefaultInitConfigNormal()
	if cfg == nil {
		return base
	}
	if cfg.Distribution == initializer.DistributionUniform || cfg.Distribution == initializer.DistributionNormal {
		base = DefaultInitConfig(cfg.Distribution)
	}
	if cfg.Conv != nil {
		base.Conv = cfg.Conv
	}
	if cfg.Linear != nil {
		base.Linear = cfg.Linear
	}
	if cfg.Attention != nil {
		base.Attention = cfg.Attention
	}
	if cfg.FFN != nil {
		base.FFN = cfg.FFN
	}
	return base
}

var (
	noopWeights = func(_ *num.Data) {}
	noopTriple  = func(_, _, _ *num.Data) {}
)

// NewTransformerBlock builds a single transformer block:
// Residual(RMSNorm -> MHA -> Dropout) + Residual(RMSNorm -> SwiGLU -> Dropout).
func NewTransformerBlock(
	featuresCount int,
	headSize int,
	headsCount int,
	contextLength int,
	hiddenSize int,
	dropoutProb float32,
	initCfg *InitConfig,
) layer.Layers {
	cfg := normalizeInitConfig(initCfg)

	return layer.Layers{
		layer.NewResidual(layer.Layers{
			layer.NewRMSLNorm(),
			layer.NewSAMultiHead(featuresCount, headSize, headsCount, contextLength, cfg.Attention, noopTriple),
			layer.NewDropout(dropoutProb),
		}),
		layer.NewResidual(layer.Layers{
			layer.NewRMSLNorm(),
			layer.NewSwiGLU(featuresCount, hiddenSize, cfg.FFN, noopTriple),
			layer.NewDropout(dropoutProb),
		}),
	}
}

// NewConvFeatureExtractor builds a simple stack: [Conv -> ReLU] * N.
func NewConvFeatureExtractor(
	filterSizes []int,
	filtersCounts []int,
	batchSize int,
	padding int,
	stride int,
	initCfg *InitConfig,
) layer.Layers {
	if len(filterSizes) != len(filtersCounts) {
		panic("filterSizes and filtersCounts must have the same length")
	}
	cfg := normalizeInitConfig(initCfg)

	layers := make(layer.Layers, 0, len(filterSizes)*2)
	for i := range filterSizes {
		layers = append(layers,
			layer.NewConv(filterSizes[i], filtersCounts[i], batchSize, padding, stride, cfg.Conv, noopWeights),
			layer.NewReLu(),
		)
	}
	return layers
}
