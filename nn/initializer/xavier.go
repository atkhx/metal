package initializer

import "math"

var (
	XavierNormalReLU   = XavierNormal{gain: reLuGain}
	XavierNormalLinear = XavierNormal{gain: linearGain}
	XavierUniformReLU  = XavierUniform{gain: reLuGain}
)

type XavierNormal struct {
	gain float32
}

func (wi XavierNormal) GetNormK(fanIn, fanOut int) float32 {
	return wi.gain * float32(math.Sqrt(2/float64(fanIn+fanOut)))
}

type XavierUniform struct {
	gain float32
}

func (wi XavierUniform) GetNormK(fanIn, fanOut int) float32 {
	return wi.gain * float32(math.Sqrt(6/float64(fanIn+fanOut)))
}
