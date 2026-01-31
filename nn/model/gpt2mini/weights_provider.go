package gpt2mini

import "github.com/atkhx/metal/nn/num"

type (
	WeightsProvider struct {
		WeightsSTReader
	}
)

func (w *WeightsProvider) copyWithCheckLength(dst *num.Data, src []float32) {
	if len(dst.Data.GetFloats()) != len(src) {
		panic("mismatching src and dst length")
	}
	copy(dst.Data.GetFloats(), src)
}

func (w *WeightsProvider) ProvideWTE(dst *num.Data) {
	w.copyWithCheckLength(dst, w.Must(w.ReadWTE()))
}

func (w *WeightsProvider) ProvideWPE(dst *num.Data) {
	w.copyWithCheckLength(dst, w.Must(w.ReadWPE()))
}

func (w *WeightsProvider) ProvideBlockLN1(block int) func(gamma, beta *num.Data) {
	return func(gamma, beta *num.Data) {
		w.copyWithCheckLength(gamma, w.Must(w.ReadBlockLN1Weights(block)))
		w.copyWithCheckLength(beta, w.Must(w.ReadBlockLN1Bias(block)))
	}
}

func (w *WeightsProvider) ProvideBlockLN2(block int) func(gamma, beta *num.Data) {
	return func(gamma, beta *num.Data) {
		w.copyWithCheckLength(gamma, w.Must(w.ReadBlockLN2Weights(block)))
		w.copyWithCheckLength(beta, w.Must(w.ReadBlockLN2Bias(block)))
	}
}

func (w *WeightsProvider) ProvideBlockQKV(block int) func(qw, kw, vw, qb, kb, vb *num.Data) {
	// TODO try to build model without splitting qkv
	return func(qw, kw, vw, qb, kb, vb *num.Data) {
		// read separate qkv weights
		q, k, v, err := w.ReadSeparateBlockQKVWeights(block)
		if err != nil {
			panic(err)
		}
		w.copyWithCheckLength(qw, q)
		w.copyWithCheckLength(kw, k)
		w.copyWithCheckLength(vw, v)
		// read separate bias
		q, k, v, err = w.ReadSeparateBlockQKVBias(block)
		if err != nil {
			panic(err)
		}
		w.copyWithCheckLength(qb, q)
		w.copyWithCheckLength(kb, k)
		w.copyWithCheckLength(vb, v)
	}
}

func (w *WeightsProvider) ProvideBlockAttnProj(block int) func(weights, bias *num.Data) {
	return func(weights, bias *num.Data) {
		w.copyWithCheckLength(weights, w.Must(w.ReadBlockAttnProjWeights(block)))
		w.copyWithCheckLength(bias, w.Must(w.ReadBlockAttnProjBias(block)))
	}
}

func (w *WeightsProvider) ProvideBlockMLPFC(block int) func(weights, bias *num.Data) {
	return func(weights, bias *num.Data) {
		w.copyWithCheckLength(weights, w.Must(w.ReadBlockMLPFCWeights(block)))
		w.copyWithCheckLength(bias, w.Must(w.ReadBlockMLPFCBias(block)))
	}
}

func (w *WeightsProvider) ProvideBlockMLPProj(block int) func(weights, bias *num.Data) {
	return func(weights, bias *num.Data) {
		w.copyWithCheckLength(weights, w.Must(w.ReadBlockMLPProjWeights(block)))
		w.copyWithCheckLength(bias, w.Must(w.ReadBlockMLPProjBias(block)))
	}
}

func (w *WeightsProvider) ProvideFinalLN() func(gamma, beta *num.Data) {
	return func(gamma, beta *num.Data) {
		w.copyWithCheckLength(gamma, w.Must(w.ReadFinalLNWeights()))
		w.copyWithCheckLength(beta, w.Must(w.ReadFinalLNBias()))
	}
}
