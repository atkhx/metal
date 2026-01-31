package gpt2

type Config struct {
	ContextLength   int
	FeaturesCount   int
	HeadsCount      int
	HeadSize        int
	HiddenDim       int
	BlocksCount     int
	VocabSize       int
	BatchSize       int
	DropoutProb     float32
	LayerNormEps    float32
	WeightsProvider *WeightsProvider
}

func GetDefaultConfig() Config {
	return Config{
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
}
