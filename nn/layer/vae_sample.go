package layer

import (
	"fmt"

	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

// VAESample reparameterizes (mu, logvar) into z = mu + exp(0.5 * logvar) * eps.
type VAESample struct {
	latentDim int
	output    *num.Data
}

func NewVAESample(latentDim int) *VAESample {
	if latentDim < 1 {
		panic("VAESample: latentDim must be >= 1")
	}
	return &VAESample{latentDim: latentDim}
}

func (l *VAESample) Compile(device *proc.Device, input *num.Data) *num.Data {
	if input.Dims.W != l.latentDim*2 {
		panic(fmt.Sprintf("VAESample: input width %d must be 2*latentDim (%d)", input.Dims.W, l.latentDim*2))
	}
	l.output = device.VAESample(input, l.latentDim)
	return l.output
}

func (l *VAESample) GetOutput() *num.Data {
	return l.output
}
