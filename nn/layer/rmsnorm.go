package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewRMSLNorm() *RMSNorm {
	return &RMSNorm{}
}

type RMSNorm struct{}

func (l *RMSNorm) Compile(device *proc.Device, input *num.Data) *num.Data {
	return device.RMSNorm(input, input.Dims.W)
}
