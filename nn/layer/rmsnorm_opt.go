package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewRMSNormOpt() *RMSNormOpt {
	return &RMSNormOpt{}
}

type RMSNormOpt struct{}

func (l *RMSNormOpt) Compile(device *proc.Device, input *num.Data) *num.Data {
	return device.RMSNormOpt(input, input.Dims.W)
}
