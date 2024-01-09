package initializer

import "math"

const (
	reLuGain    = 1.4142135624 // √2
	tanhGain    = 1.6666666667 // 5/3
	sigmoidGain = 1.0
	linearGain  = 1.0
)

var (
	KaimingNormalReLU    = KaimingNormal{gain: reLuGain}
	KaimingNormalTanh    = KaimingNormal{gain: tanhGain}
	KaimingNormalSigmoid = KaimingNormal{gain: sigmoidGain}
	KaimingNormalLinear  = KaimingNormal{gain: linearGain}
)

type Initializer interface {
	GetNormK(fanIn int) float32
}

type InitWeightFixed struct {
	NormK float32
}

func (wi InitWeightFixed) GetNormK(fanIn int) float32 {
	return wi.NormK
}

type KaimingNormal struct {
	gain float32
}

func (wi KaimingNormal) GetNormK(fanIn int) float32 {
	return wi.gain / float32(math.Sqrt(float64(fanIn)))
}
