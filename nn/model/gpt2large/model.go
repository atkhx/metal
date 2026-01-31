package gpt2large

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/model"
	"github.com/atkhx/metal/nn/proc"

	jsoniter "github.com/json-iterator/go"
)

var json = jsoniter.ConfigCompatibleWithStandardLibrary

func NewModel(cfg Config, device *proc.Device, optimizer proc.Optimizer) *model.Model {
	inDims := mtl.MTLSize{
		W: cfg.ContextLength,
		H: cfg.BatchSize,
		D: 1,
	}

	provider := cfg.WeightsProvider

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
				layer.NewLinear(cfg.FeaturesCount, initializer.XavierNormalLinear, true, provider.ProvideBlockAttnProj(block)),
				layer.NewDropout(cfg.DropoutProb),
			}),
			layer.NewResidual(layer.Layers{
				layer.NewLayerNormAffine(cfg.FeaturesCount, cfg.LayerNormEps, provider.ProvideBlockLN2(block)),
				layer.NewLinear(cfg.HiddenDim, initializer.KaimingNormalReLU, true, provider.ProvideBlockMLPFC(block)),
				layer.NewGeLu(),
				//layer.NewGeLuNew(),
				layer.NewLinear(cfg.FeaturesCount, initializer.KaimingNormalReLU, true, provider.ProvideBlockMLPProj(block)),
				layer.NewDropout(cfg.DropoutProb),
			}),
		)
	}

	layers = append(layers,
		layer.NewLayerNormAffine(cfg.FeaturesCount, cfg.LayerNormEps, provider.ProvideFinalLN()),
		layer.NewLinearWithImmutableWeights(embeddingsOut),
		layer.NewReshape(mtl.NewMTLSize(cfg.VocabSize, cfg.BatchSize*cfg.ContextLength)),
	)

	return model.New(inDims, layers, device, optimizer)
}
