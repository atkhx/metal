package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewReLu() *Activation {
	return &Activation{activation: func(device *proc.Device, input *num.Data) *num.Data {
		return device.Relu(input)
	}}
}

type Activation struct {
	activation func(device *proc.Device, input *num.Data) *num.Data
}

func (l *Activation) Compile(device *proc.Device, input *num.Data) *num.Data {
	return l.activation(device, input)
}
