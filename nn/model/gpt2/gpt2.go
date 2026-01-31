package model

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/proc"
)

func GetDefaultGPT2Config() GPT2Config {
	return GPT2Config{
		ContextLength: 512,
		FeaturesCount: 512,
		HeadsCount:    8,
		HeadSize:      64,
		HiddenDim:     2048,
		BlocksCount:   4,
		VocabSize:     50257,
		BatchSize:     1,
		DropoutProb:   0,
		LayerNormEps:  1e-5,
	}
}

type GPT2Config struct {
	ContextLength int
	FeaturesCount int
	HeadsCount    int
	HeadSize      int
	HiddenDim     int
	BlocksCount   int
	VocabSize     int
	BatchSize     int
	DropoutProb   float32
	LayerNormEps  float32
	Weights       *GPT2Weights
}

func NewGPT2(cfg GPT2Config, device *proc.Device, optimizer proc.Optimizer) *Model {
	inDims := mtl.MTLSize{
		W: cfg.ContextLength,
		H: cfg.BatchSize,
		D: 1,
	}

	weights := cfg.Weights
	provider := GPT2WeightsProvider{GPT2Weights: *weights}

	embeddingsIn := device.NewTokenEmbeddingTable(
		cfg.FeaturesCount,
		cfg.VocabSize,
		initializer.XavierNormalLinear.GetNormK(cfg.FeaturesCount, cfg.VocabSize),
	)
	embeddingsOut := device.Transpose(embeddingsIn)

	layers := layer.Layers{
		layer.NewEmbeddings(embeddingsIn, provider.ProvideWTE),
		layer.NewPositionalAdd(cfg.ContextLength, cfg.FeaturesCount, provider.ProvideWPE),
	}
	for i := 0; i < cfg.BlocksCount; i++ {
		block := i
		layers = append(layers,
			layer.NewResidual(layer.Layers{
				layer.NewLayerNormAffine(cfg.FeaturesCount, cfg.LayerNormEps, provider.ProvideBlockLN1(block)),
				layer.NewSAMultiHeadWithBias(
					cfg.FeaturesCount,
					cfg.HeadSize,
					cfg.HeadsCount,
					cfg.ContextLength,
					false,
					initializer.XavierNormalLinear,
					provider.ProvideBlockQKV(block),
				),
				layer.NewLinear(cfg.FeaturesCount, initializer.XavierNormalLinear, true, provider.ProvideBlockProj(block)),
				layer.NewDropout(cfg.DropoutProb),
			}),
			layer.NewResidual(layer.Layers{
				layer.NewLayerNormAffine(cfg.FeaturesCount, cfg.LayerNormEps, provider.ProvideBlockLN2(block)),
				layer.NewLinear(cfg.HiddenDim, initializer.KaimingNormalReLU, true, provider.ProvideBlockMLP1(block)),
				layer.NewGeLu(),
				//layer.NewGeLuNew(),
				layer.NewLinear(cfg.FeaturesCount, initializer.KaimingNormalReLU, true, provider.ProvideBlockMLP2(block)),
				layer.NewDropout(cfg.DropoutProb),
			}),
		)
	}

	layers = append(layers,
		layer.NewLayerNormAffine(cfg.FeaturesCount, cfg.LayerNormEps, provider.ProvideFinalLN()),
		layer.NewLinearWithImmutableWeights(embeddingsOut),
		layer.NewReshape(mtl.NewMTLSize(cfg.VocabSize, cfg.BatchSize*cfg.ContextLength)),
	)

	return New(inDims, layers, device, optimizer)
}
