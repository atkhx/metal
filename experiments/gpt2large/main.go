package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	"github.com/atkhx/metal/nn/model/gpt2large"
	"github.com/atkhx/metal/nn/model/safetensors"
	tokenizergpt2bpe "github.com/atkhx/metal/nn/model/tokenizer_gpt2_bpe"
	"github.com/atkhx/metal/nn/proc"
)

var (
	weightsFile = flag.String("weights", "data/gpt2large/model.safetensors", "weights file")
	configPath  = flag.String("config", "data/gpt2large/gpt2-large/config.json", "gpt2 config.json")
	vocabPath   = flag.String("vocab", "data/gpt2large/gpt2-large/vocab.json", "vocab.json for decode")
	mergesPath  = flag.String("merges", "data/gpt2large/gpt2-large/merges.txt", "merges.txt for encode")
	prompt      = flag.String("prompt", "", "prompt text (overrides -tokens)")
	steps       = flag.Int("steps", 1024, "generation steps")
)

func main() {
	flag.Parse()

	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	cfg, err := gpt2large.LoadHFConfig(*configPath, 1, 0)
	if err != nil {
		log.Printf("config load failed (%v), using defaults", err)
		cfg = gpt2large.GetDefaultConfig()
	}

	tokenizer, err := tokenizergpt2bpe.NewFromFiles(*vocabPath, *mergesPath)
	if err != nil {
		log.Fatalf("init tokenizer: %v", err)
	}

	inputTokens, err := tokenizer.Encode(*prompt)
	if err != nil {
		log.Fatalf("encode prompt: %v", err)
	}

	weights, err := os.Open(*weightsFile)
	if err != nil {
		log.Fatalf("open weights file: %v", err)
	}
	defer weights.Close()

	weightsReader, err := safetensors.NewReader(weights)
	if err != nil {
		log.Fatalf("create reader: %v", err)
	}

	cfg.WeightsProvider = &gpt2large.WeightsProvider{
		WeightsSTReader: gpt2large.WeightsSTReader{WeightsReader: weightsReader},
	}

	m := gpt2large.NewModel(cfg, device, nil)
	m.Compile()
	m.LoadFromProvider()

	output := m.GetOutput()
	pipeline := device.GetInferencePipeline(output)

	fmt.Println()
	fmt.Print(*prompt)
	for i := 0; i < *steps; i++ {
		//inputFloats, pos := prepareInputFloatsPost(inputTokens, cfg.ContextLength)
		inputFloats, pos := prepareInputFloatsPre(inputTokens, cfg.ContextLength)
		copy(m.GetInput().Data.GetFloats(), inputFloats)
		pipeline.Forward()

		logits := output.Data.GetFloats()[pos*cfg.VocabSize : (pos+1)*cfg.VocabSize]
		next := sampleWithTemperature(logits, 0.9)

		inputTokens = append(inputTokens, uint32(next))
		if len(inputTokens) > cfg.ContextLength {
			inputTokens = inputTokens[len(inputTokens)-cfg.ContextLength:]
		}
		if s, err := tokenizer.Decode([]uint32{uint32(next)}); err != nil {
			log.Fatalf("decode tokens: %v", err)
		} else {
			fmt.Print(s)
		}
	}
	fmt.Println()
}

func prepareInputFloatsPost(inputTokens []uint32, contextLength int) ([]float32, int) {
	result := make([]float32, contextLength)
	for i := range result {
		result[i] = 50256
		//result[i] = 0
	}
	offset := contextLength - len(inputTokens)
	for i, token := range inputTokens {
		result[offset+i] = float32(token)
	}
	return result, contextLength - 1
}

func prepareInputFloatsPre(inputTokens []uint32, contextLength int) ([]float32, int) {
	result := make([]float32, contextLength)
	for i := range result {
		//result[i] = 50256
		result[i] = 0
	}
	for i, token := range inputTokens {
		result[i] = float32(token)
	}
	return result, len(inputTokens) - 1
}

func sampleWithTemperature(logits []float32, temperature float32) int {
	for i := 0; i < len(logits); i++ {
		logits[i] /= temperature
	}
	return sampleFromDistribution(softmax(logits))
}

func softmax(logits []float32) []float32 {
	probs := make([]float32, len(logits))
	sum := 0.0
	maxLogit := float32(math.Inf(-1))

	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	for _, logit := range logits {
		sum += math.Exp(float64(logit - maxLogit))
	}

	for i, logit := range logits {
		probs[i] = float32(math.Exp(float64(logit-maxLogit)) / sum)
	}
	return probs
}

func sampleFromDistribution(probs []float32) int {
	r := float32(rand.Float64())
	cumulativeProb := float32(0.0)

	for i, prob := range probs {
		cumulativeProb += prob
		if r <= cumulativeProb {
			return i
		}
	}
	return len(probs) - 1
}
