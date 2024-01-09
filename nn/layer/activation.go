package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewReLu() *Activation {
	return &Activation{inputsAct: func(device *proc.Device, inputs *num.Data) *num.Data {
		return device.Relu(inputs)
	}}
}

type Activation struct {
	inputsAct func(device *proc.Device, inputs *num.Data) *num.Data
}

func (l *Activation) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	return l.inputsAct(device, inputs)
}
