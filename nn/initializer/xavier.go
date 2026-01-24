package initializer

import "math"

var (
	XavierUniformReLU   = XavierUniform{gain: reLuGain}
	XavierUniformLinear = XavierUniform{gain: linearGain}

	XavierNormalReLU   = XavierNormal{gain: reLuGain}
	XavierNormalLinear = XavierNormal{gain: linearGain}
)

// XavierUniform uses U(-a, a), where a = gain * sqrt(6/(fanIn+fanOut)).
type XavierUniform struct {
	gain float32
}

func (wi XavierUniform) GetNormK(fanIn, fanOut int) float32 {
	return wi.gain * float32(math.Sqrt(6/float64(fanIn+fanOut)))
}

func (wi XavierUniform) Distribution() Distribution {
	return DistributionUniform
}

// XavierNormal uses N(0, std^2), where std = gain * sqrt(2/(fanIn+fanOut)).
type XavierNormal struct {
	gain float32
}

func (wi XavierNormal) GetNormK(fanIn, fanOut int) float32 {
	return wi.gain * float32(math.Sqrt(2/float64(fanIn+fanOut)))
}

func (wi XavierNormal) Distribution() Distribution {
	return DistributionNormal
}
