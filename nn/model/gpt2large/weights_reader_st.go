package gpt2large

import (
	"fmt"
)

type (
	WeightsReader interface {
		ReadTensor(name string) ([]float32, error)
		TensorShape(name string) ([]int, error)
	}
	WeightsSTReader struct {
		WeightsReader
	}
)

func (w *WeightsSTReader) Must(f []float32, err error) []float32 {
	if err != nil {
		panic(err)
	}
	return f
}

func (w *WeightsSTReader) ReadWTE() ([]float32, error) {
	return w.ReadTensor("wte.weight")
}

func (w *WeightsSTReader) ReadWPE() ([]float32, error) {
	return w.ReadTensor("wpe.weight")
}

func (w *WeightsSTReader) ReadBlockLN1Weights(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.ln_1.weight", block))
}

func (w *WeightsSTReader) ReadBlockLN1Bias(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.ln_1.bias", block))
}

func (w *WeightsSTReader) ReadBlockLN2Weights(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.ln_2.weight", block))
}

func (w *WeightsSTReader) ReadBlockLN2Bias(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.ln_2.bias", block))
}

func (w *WeightsSTReader) ReadBlockQKVWeights(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.attn.c_attn.weight", block))
}

func (w *WeightsSTReader) ReadBlockQKVBias(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.attn.c_attn.bias", block))
}

func (w *WeightsSTReader) ReadSeparateBlockQKVWeights(block int) ([]float32, []float32, []float32, error) {
	data, err := w.ReadBlockQKVWeights(block)
	if err != nil {
		return nil, nil, nil, err
	}
	shape, err := w.TensorShape(fmt.Sprintf("h.%d.attn.c_attn.weight", block))
	if err != nil {
		return nil, nil, nil, err
	}
	if len(shape) != 2 {
		return nil, nil, nil, fmt.Errorf("expected two shapes")
	}
	rows, cols := shape[0], shape[1]
	size := rows * rows
	data = w.transpose(data, cols, rows)
	return data[:size], data[size : 2*size], data[2*size:], nil
}

func (w *WeightsSTReader) transpose(src []float32, cols, rows int) []float32 {
	out := make([]float32, cols*rows)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			out[c*rows+r] = src[r*cols+c]
		}
	}
	return out
}

func (w *WeightsSTReader) ReadSeparateBlockQKVBias(block int) ([]float32, []float32, []float32, error) {
	b, err := w.ReadBlockQKVBias(block)
	if err != nil {
		return nil, nil, nil, err
	}
	chunkSize, rem := len(b)/3, len(b)%3
	if rem != 0 {
		return nil, nil, nil, fmt.Errorf("invalid block size: %d", len(b))
	}
	return b[:chunkSize], b[chunkSize : 2*chunkSize], b[2*chunkSize:], nil
}

func (w *WeightsSTReader) ReadBlockAttnProjWeights(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.attn.c_proj.weight", block))
}

func (w *WeightsSTReader) ReadBlockAttnProjBias(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.attn.c_proj.bias", block))
}

func (w *WeightsSTReader) ReadBlockMLPProjWeights(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.mlp.c_proj.weight", block))
}

func (w *WeightsSTReader) ReadBlockMLPProjBias(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.mlp.c_proj.bias", block))
}

func (w *WeightsSTReader) ReadBlockMLPFCWeights(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.mlp.c_fc.weight", block))
}

func (w *WeightsSTReader) ReadBlockMLPFCBias(block int) ([]float32, error) {
	return w.ReadTensor(fmt.Sprintf("h.%d.mlp.c_fc.bias", block))
}

func (w *WeightsSTReader) ReadFinalLNWeights() ([]float32, error) {
	return w.ReadTensor("ln_f.weight")
}

func (w *WeightsSTReader) ReadFinalLNBias() ([]float32, error) {
	return w.ReadTensor("ln_f.bias")
}
