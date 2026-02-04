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

func NewGeLu() *Activation {
	return &Activation{activation: func(device *proc.Device, input *num.Data) *num.Data {
		return device.GeLu(input)
	}}
}

func NewGeLuNew() *Activation {
	return &Activation{activation: func(device *proc.Device, input *num.Data) *num.Data {
		return device.GeLuNew(input)
	}}
}

func NewSigmoid() *Activation {
	return &Activation{activation: func(device *proc.Device, input *num.Data) *num.Data {
		return device.Sigmoid(input)
	}}
}

type Activation struct {
	activation func(device *proc.Device, input *num.Data) *num.Data
}

func (l *Activation) Compile(device *proc.Device, input *num.Data) *num.Data {
	return l.activation(device, input)
}
