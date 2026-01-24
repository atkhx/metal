package model

import (
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
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
			layer.NewConv(filterSizes[i], filtersCounts[i], batchSize, padding, stride, cfg.Conv, nil),
			layer.NewReLu(),
		)
	}
	return layers
}

// NewConvNet builds a Conv feature extractor followed by an optional Linear head.
// The head is only added when headFeatures > 0.
func NewConvNet(
	filterSizes []int,
	filtersCounts []int,
	batchSize int,
	padding int,
	stride int,
	headFeatures int,
	initCfg *InitConfig,
) layer.Layers {
	cfg := normalizeInitConfig(initCfg)
	layers := NewConvFeatureExtractor(filterSizes, filtersCounts, batchSize, padding, stride, &cfg)
	if headFeatures > 0 {
		layers = append(layers, layer.NewLinear(headFeatures, cfg.Linear, true, nil))
	}
	return layers
}
