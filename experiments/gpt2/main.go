package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/atkhx/metal/nn/model/gpt2"
	"github.com/atkhx/metal/nn/model/safetensors"
	tokenizergpt2bpe "github.com/atkhx/metal/nn/model/tokenizer_gpt2_bpe"
	"github.com/atkhx/metal/nn/proc"
)

var (
	weightsFile = flag.String("weights", "data/gpt2%s/model.safetensors", "gpt2 model.safetensors")
	configPath  = flag.String("config", "data/gpt2%s/config.json", "gpt2 config.json")
	vocabPath   = flag.String("vocab", "data/gpt2%s/vocab.json", "tokenizer vocab.json")
	mergesPath  = flag.String("merges", "data/gpt2%s/merges.txt", "tokenizer merges.txt")

	prompt = flag.String("prompt", "Hi!", "prompt text")
	model  = flag.String("model", "mini", "which model to use (mini, medium or large)")

	steps = flag.Int("steps", 2048, "generation steps")
	temp  = flag.Float64("temp", 0.9, "sampling temperature")
	topK  = flag.Int("topk", 0, "top-k sampling (0 disables)")
	topP  = flag.Float64("topp", 0.0, "top-p sampling (0 disables)")
)

func main() {
	var err error
	defer func() {
		if err != nil {
			fmt.Println("script failed")
			log.Fatalln(err)
		}
	}()

	flag.Parse()

	modelType, err := gpt2.ModelTypeFromString(*model)
	if err != nil {
		err = fmt.Errorf("parse model type %s: %w", *model, err)
		return
	}

	*weightsFile = fmt.Sprintf(*weightsFile, *model)
	*configPath = fmt.Sprintf(*configPath, *model)
	*vocabPath = fmt.Sprintf(*vocabPath, *model)
	*mergesPath = fmt.Sprintf(*mergesPath, *model)

	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	cfg, err := gpt2.LoadHFConfig(*configPath, 1, 0)
	if err != nil {
		log.Printf("config load failed '%v' (%w), using defaults", *configPath, err)
		if cfg, err = gpt2.GetDefaultConfig(modelType); err != nil {
			err = fmt.Errorf("get default config: %w", err)
			return
		}
	}

	tokenizer, err := tokenizergpt2bpe.NewFromFiles(*vocabPath, *mergesPath)
	if err != nil {
		err = fmt.Errorf("create tokenizer: %w", err)
		return
	}

	weights, err := os.Open(*weightsFile)
	if err != nil {
		err = fmt.Errorf("open weights file: %w", err)
		return
	}
	defer weights.Close()

	weightsReader, err := safetensors.NewReader(weights)
	if err != nil {
		err = fmt.Errorf("create safetensors weights reader: %w", err)
		return
	}

	cfg.WeightsProvider = &gpt2.WeightsProvider{
		WeightsSTReader: gpt2.WeightsSTReader{
			WeightsReader: weightsReader,
			WeightsPrefix: gpt2.GetSTWeightPrefix(modelType),
		},
	}

	inputFloats, err := tokenizer.EncodeToFloat32s(*prompt, cfg.ContextLength)
	if err != nil {
		err = fmt.Errorf("encode prompt: %w", err)
		return
	}

	input, output, pipeline := gpt2.NewModelForTest(cfg, device)

	runSteps := func(onGetNextToken func(token string)) error {
		for i := 0; i < *steps; i++ {
			copy(input.Data.GetFloats(), inputFloats)
			pipeline.Forward()

			pos := len(inputFloats) - 1
			logits := output.Data.GetFloats()[pos*cfg.VocabSize : (pos+1)*cfg.VocabSize]
			next := gpt2.SampleNextTokenWithConfig(logits, gpt2.SamplerConfig{
				Temperature: float32(*temp),
				TopK:        *topK,
				TopP:        float32(*topP),
			})

			inputFloats = append(inputFloats, float32(next))
			if len(inputFloats) > cfg.ContextLength {
				inputFloats = inputFloats[len(inputFloats)-cfg.ContextLength:]
			}

			s, err := tokenizer.Decode([]uint32{uint32(next)})
			if err != nil {
				return fmt.Errorf("tokenizer decode: %v", err)
			}
			onGetNextToken(s)

			if uint32(next) == tokenizergpt2bpe.TokenEOT {
				break
			}
		}
		return nil
	}

	fmt.Println()
	fmt.Print(*prompt)
	if err = runSteps(func(token string) {
		fmt.Print(token)
	}); err != nil {
		return
	}
	fmt.Println()
}
