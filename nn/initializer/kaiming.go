package initializer

import "math"

var (
	KaimingUniformReLU    = KaimingUniform{gain: reLuGain}
	KaimingUniformTanh    = KaimingUniform{gain: tanhGain}
	KaimingUniformSigmoid = KaimingUniform{gain: sigmoidGain}
	KaimingUniformLinear  = KaimingUniform{gain: linearGain}

	KaimingNormalReLU    = KaimingNormal{gain: reLuGain}
	KaimingNormalTanh    = KaimingNormal{gain: tanhGain}
	KaimingNormalSigmoid = KaimingNormal{gain: sigmoidGain}
	KaimingNormalLinear  = KaimingNormal{gain: linearGain}
)

// KaimingUniform uses U(-a, a), where a = gain * sqrt(6/fanIn).
type KaimingUniform struct {
	gain float32
}

func (wi KaimingUniform) GetNormK(fanIn, _ int) float32 {
	return wi.gain * float32(math.Sqrt(6/float64(fanIn)))
}

func (wi KaimingUniform) Distribution() Distribution {
	return DistributionUniform
}

// KaimingNormal uses N(0, std^2), where std = gain * sqrt(2/fanIn).
type KaimingNormal struct {
	gain float32
}

func (wi KaimingNormal) GetNormK(fanIn, _ int) float32 {
	return wi.gain * float32(math.Sqrt(2/float64(fanIn)))
}

func (wi KaimingNormal) Distribution() Distribution {
	return DistributionNormal
}
