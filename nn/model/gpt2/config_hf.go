package gpt2

import (
	"fmt"
	"os"
)

type hfGPT2Config struct {
	NEmb         int     `json:"n_embd"`
	NHead        int     `json:"n_head"`
	NInner       int     `json:"n_inner"`
	NLayer       int     `json:"n_layer"`
	NPositions   int     `json:"n_positions"`
	VocabSize    int     `json:"vocab_size"`
	LayerNormEps float32 `json:"layer_norm_epsilon"`
}

func LoadGPT2Config(path string, batchSize int, dropout float32) (GPT2Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return GPT2Config{}, fmt.Errorf("read config: %w", err)
	}
	var cfg hfGPT2Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return GPT2Config{}, fmt.Errorf("parse config: %w", err)
	}
	if cfg.NEmb == 0 || cfg.NHead == 0 || cfg.NPositions == 0 || cfg.VocabSize == 0 {
		return GPT2Config{}, fmt.Errorf("invalid config: %+v", cfg)
	}
	headSize := cfg.NEmb / cfg.NHead
	return GPT2Config{
		ContextLength: cfg.NPositions,
		FeaturesCount: cfg.NEmb,
		HeadsCount:    cfg.NHead,
		HeadSize:      headSize,
		HiddenDim:     cfg.NInner,
		BlocksCount:   cfg.NLayer,
		VocabSize:     cfg.VocabSize,
		BatchSize:     batchSize,
		DropoutProb:   dropout,
		LayerNormEps:  defaultEps(cfg.LayerNormEps),
	}, nil
}

func defaultEps(eps float32) float32 {
	if eps == 0 {
		return 1e-5
	}
	return eps
}
