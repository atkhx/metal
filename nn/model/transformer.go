package model

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/proc"
)

type TransformerBlockInit struct {
	Attention initializer.Initializer
	Linear    initializer.Initializer
	FFN       initializer.Initializer
	RMSMul    initializer.Initializer
}

type TransformerBlockConfig struct {
	FeaturesCount int
	HeadSize      int
	HeadsCount    int
	ContextLength int
	HiddenSize    int

	DropoutProb float32

	UseRMSNorm bool
	UseMulRows bool
	UseDropout bool
}

// NewTransformer builds a GPT-style LM with tied input/output embeddings.
// Output is reshaped to [alphabetSize, contextLength*batchSize] for CrossEntropyPos.
func NewTransformer(
	contextLength int,
	embeddingFeatures int,
	headsCount int,
	headSize int,
	headLinearSize int,
	blocksCount int,
	alphabetSize int,
	miniBatchSize int,
	dropout float32,
	withHead bool,
	initCfg *InitConfig,
	device *proc.Device,
	modelOptimizer proc.Optimizer,
) *Model {
	cfg := normalizeInitConfig(initCfg)

	inDims := mtl.MTLSize{
		W: contextLength,
		H: miniBatchSize,
		D: 1,
	}

	initWeightRMSMul := &initializer.InitWeightFixed{NormK: 1}

	layers := layer.Layers{}

	embeddingsIn := device.NewTokenEmbeddingTable(embeddingFeatures, alphabetSize, cfg.Linear.GetNormK(embeddingFeatures, alphabetSize))
	embeddingsOut := device.Transpose(embeddingsIn)

	layers = append(layers, layer.NewEmbeddings(embeddingsIn, nil))

	blockCfg := TransformerBlockConfig{
		FeaturesCount: embeddingFeatures,
		HeadSize:      headSize,
		HeadsCount:    headsCount,
		ContextLength: contextLength,
		HiddenSize:    headLinearSize,
		DropoutProb:   dropout,
		UseRMSNorm:    true,
		UseMulRows:    true,
		UseDropout:    dropout > 0,
	}
	blockInit := TransformerBlockInit{
		Attention: cfg.Attention,
		Linear:    cfg.Linear,
		FFN:       cfg.FFN,
		RMSMul:    initWeightRMSMul,
	}

	for i := 0; i < blocksCount; i++ {
		layers = append(layers, NewTransformerBlock(blockCfg, blockInit)...)
	}

	if withHead {
		layers = append(layers,
			layer.NewRMSLNorm(),
			layer.NewMulRows(embeddingFeatures, initWeightRMSMul, nil),
			layer.NewLinearWithImmutableWeights(embeddingsOut),
		)

		layers = append(layers, layer.NewReshape(mtl.NewMTLSize(alphabetSize, miniBatchSize*contextLength)))
	}

	return New(inDims, layers, device, modelOptimizer)
}

func NewTransformerBlock(
	cfg TransformerBlockConfig,
	init TransformerBlockInit,
) layer.Layers {
	init.Attention = initializer.DefaultOnNil(init.Attention, initializer.XavierNormalLinear)
	init.Linear = initializer.DefaultOnNil(init.Linear, initializer.XavierNormalLinear)
	init.FFN = initializer.DefaultOnNil(init.FFN, initializer.KaimingNormalReLU)
	init.RMSMul = initializer.DefaultOnNil(init.RMSMul, initializer.InitWeightFixed{NormK: 1})

	dropProb := float32(0)
	if cfg.UseDropout {
		dropProb = cfg.DropoutProb
	}

	buildPre := func() layer.Layers {
		var pre layer.Layers
		if cfg.UseRMSNorm {
			pre = append(pre, layer.NewRMSLNorm())
		}
		if cfg.UseMulRows {
			pre = append(pre, layer.NewMulRows(cfg.FeaturesCount, init.RMSMul, nil))
		}
		return pre
	}

	attn := layer.Layers{
		layer.NewSAMultiHead(cfg.FeaturesCount, cfg.HeadSize, cfg.HeadsCount, cfg.ContextLength, init.Attention, nil),
		layer.NewLinear(cfg.FeaturesCount, init.Linear, false, nil),
	}
	if dropProb > 0 {
		attn = append(attn, layer.NewDropout(dropProb))
	}

	ffn := layer.Layers{layer.NewSwiGLU(cfg.FeaturesCount, cfg.HiddenSize, init.FFN, nil)}
	if dropProb > 0 {
		ffn = append(ffn, layer.NewDropout(dropProb))
	}

	return layer.Layers{
		layer.NewResidual(append(buildPre(), attn...)),
		layer.NewResidual(append(buildPre(), ffn...)),
	}
}
