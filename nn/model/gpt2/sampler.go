package gpt2

import (
	"math"
	"math/rand"
	"sort"
)

type SamplerConfig struct {
	Temperature float32
	TopK        int
	TopP        float32
}

// SampleNextToken selects the next token index from logits using temperature + optional top-k/top-p.
func SampleNextToken(logits []float32, temperature float32) int {
	return SampleNextTokenWithConfig(logits, SamplerConfig{Temperature: temperature})
}

// SampleNextTokenWithConfig selects the next token index from logits using temperature + optional top-k/top-p.
func SampleNextTokenWithConfig(logits []float32, cfg SamplerConfig) int {
	filtered := applyTopKTopP(logits, cfg.TopK, cfg.TopP)
	return sampleFromDistribution(softmaxWithTemperature(filtered, cfg.Temperature))
}

func softmaxWithTemperature(logits []float32, temperature float32) []float32 {
	probs := make([]float32, len(logits))
	if len(logits) == 0 {
		return probs
	}
	if temperature <= 0 {
		temperature = 1
	}

	maxLogit := float32(math.Inf(-1))
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	var sum float64
	for _, logit := range logits {
		sum += math.Exp(float64((logit - maxLogit) / temperature))
	}

	for i, logit := range logits {
		probs[i] = float32(math.Exp(float64((logit-maxLogit)/temperature)) / sum)
	}
	return probs
}

func applyTopKTopP(logits []float32, topK int, topP float32) []float32 {
	if len(logits) == 0 {
		return logits
	}

	type pair struct {
		idx int
		val float32
	}

	items := make([]pair, len(logits))
	for i, v := range logits {
		items[i] = pair{idx: i, val: v}
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].val > items[j].val
	})

	if topK > 0 && topK < len(items) {
		items = items[:topK]
	}

	keep := items
	if topP > 0 && topP < 1 {
		sum := float64(0)
		for _, it := range items {
			sum += math.Exp(float64(it.val))
		}
		cut := 0
		cum := float64(0)
		for i, it := range items {
			cum += math.Exp(float64(it.val)) / sum
			cut = i + 1
			if cum >= float64(topP) {
				break
			}
		}
		keep = items[:cut]
	}

	filtered := make([]float32, len(logits))
	for _, it := range keep {
		filtered[it.idx] = it.val
	}
	for i := range filtered {
		if filtered[i] == 0 {
			filtered[i] = float32(math.Inf(-1))
		}
	}
	return filtered
}

func sampleFromDistribution(probs []float32) int {
	r := float32(rand.Float64())
	cumulative := float32(0.0)
	for i, prob := range probs {
		cumulative += prob
		if r <= cumulative {
			return i
		}
	}
	if len(probs) == 0 {
		return 0
	}
	return len(probs) - 1
}
