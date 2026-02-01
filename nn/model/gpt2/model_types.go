package gpt2

import "fmt"

type ModelType string

const (
	ModelTypeMini   = "mini"
	ModelTypeMedium = "medium"
	ModelTypeLarge  = "large"
)

func ModelTypeFromString(s string) (ModelType, error) {
	switch s {
	case ModelTypeMini:
		return ModelTypeMini, nil
	case ModelTypeMedium:
		return ModelTypeMedium, nil
	case ModelTypeLarge:
		return ModelTypeLarge, nil
	default:
		return "", fmt.Errorf("model type '%s' is not recognized", s)
	}
}

var (
	defaultConfigMini = Config{
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
	defaultConfigMedium = Config{
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
	defaultConfigLarge = Config{
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
)

func GetDefaultConfig(modelType ModelType) (Config, error) {
	switch modelType {
	case ModelTypeMini:
		return defaultConfigMini, nil
	case ModelTypeMedium:
		return defaultConfigMedium, nil
	case ModelTypeLarge:
		return defaultConfigLarge, nil
	default:
		return defaultConfigMini, fmt.Errorf("model type %s not supported", modelType)
	}
}

func GetSTWeightPrefix(modelType ModelType) string {
	if modelType == ModelTypeMini {
		return "transformer."
	}
	return ""
}
