package initializer

const (
	reLuGain    = 1.4142135624 // √2
	tanhGain    = 1.6666666667 // 5/3
	sigmoidGain = 1.0
	linearGain  = 1.0
)

type Initializer interface {
	GetNormK(fanIn, fanOut int) float32
}
