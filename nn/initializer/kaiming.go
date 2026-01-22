package initializer

import "math"

var (
	KaimingNormalReLU    = KaimingNormal{gain: reLuGain}
	KaimingNormalTanh    = KaimingNormal{gain: tanhGain}
	KaimingNormalSigmoid = KaimingNormal{gain: sigmoidGain}
	KaimingNormalLinear  = KaimingNormal{gain: linearGain}
)

type KaimingNormal struct {
	gain float32
}

func (wi KaimingNormal) GetNormK(fanIn, _ int) float32 {
	return wi.gain * float32(math.Sqrt(6/float64(fanIn)))
}
