package gpt2

import "github.com/atkhx/metal/nn/num"

type GPT2WeightsProvider struct {
	GPT2Weights
}

func (w *GPT2WeightsProvider) copyWithCheckLength(dst *num.Data, src []float32) {
	if len(dst.Data.GetFloats()) != len(src) {
		panic("mismatching src and dst length")
	}
	copy(dst.Data.GetFloats(), src)
}

func (w *GPT2WeightsProvider) ProvideWTE(dst *num.Data) {
	w.copyWithCheckLength(dst, w.TokenEmbedding.Data)
}

func (w *GPT2WeightsProvider) ProvideWPE(dst *num.Data) {
	w.copyWithCheckLength(dst, w.PosEmbedding.Data)
}

func (w *GPT2WeightsProvider) ProvideBlockLN1(block int) func(gamma, beta *num.Data) {
	return func(gamma, beta *num.Data) {
		w.copyWithCheckLength(gamma, w.Blocks[block].LN1Gamma)
		w.copyWithCheckLength(beta, w.Blocks[block].LN1Beta)
	}
}

func (w *GPT2WeightsProvider) ProvideBlockLN2(block int) func(gamma, beta *num.Data) {
	return func(gamma, beta *num.Data) {
		w.copyWithCheckLength(gamma, w.Blocks[block].LN2Gamma)
		w.copyWithCheckLength(beta, w.Blocks[block].LN2Beta)
	}
}

func (w *GPT2WeightsProvider) ProvideBlockQKV(block int) func(qw, kw, vw, qb, kb, vb *num.Data) {
	return func(qw, kw, vw, qb, kb, vb *num.Data) {
		w.copyWithCheckLength(qw, w.Blocks[block].Q.Data)
		w.copyWithCheckLength(kw, w.Blocks[block].K.Data)
		w.copyWithCheckLength(vw, w.Blocks[block].V.Data)
		w.copyWithCheckLength(qb, w.Blocks[block].QBias)
		w.copyWithCheckLength(kb, w.Blocks[block].KBias)
		w.copyWithCheckLength(vb, w.Blocks[block].VBias)
	}
}

func (w *GPT2WeightsProvider) ProvideBlockProj(block int) func(weights, bias *num.Data) {
	return func(weights, bias *num.Data) {
		w.copyWithCheckLength(weights, w.Blocks[block].Proj.Data)
		w.copyWithCheckLength(bias, w.Blocks[block].ProjBias)
	}
}

func (w *GPT2WeightsProvider) ProvideBlockMLP1(block int) func(weights, bias *num.Data) {
	return func(weights, bias *num.Data) {
		w.copyWithCheckLength(weights, w.Blocks[block].MLP1.Data)
		w.copyWithCheckLength(bias, w.Blocks[block].MLP1Bias)
	}
}

func (w *GPT2WeightsProvider) ProvideBlockMLP2(block int) func(weights, bias *num.Data) {
	return func(weights, bias *num.Data) {
		w.copyWithCheckLength(weights, w.Blocks[block].MLP2.Data)
		w.copyWithCheckLength(bias, w.Blocks[block].MLP2Bias)
	}
}

func (w *GPT2WeightsProvider) ProvideFinalLN() func(gamma, beta *num.Data) {
	return func(gamma, beta *num.Data) {
		w.copyWithCheckLength(gamma, w.FinalGamma)
		w.copyWithCheckLength(beta, w.FinalBeta)
	}
}
