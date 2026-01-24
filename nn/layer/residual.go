package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewResidual(layers Layers) *Residual {
	return &Residual{Layers: layers}
}

type Residual struct {
	Layers Layers
}

func (l *Residual) Compile(device *proc.Device, input *num.Data) *num.Data {
	_, output := l.Layers.Compile(device, input)
	return device.Add(input, output)
}

func (l *Residual) ForUpdate() []*num.Data {
	return l.Layers.ForUpdate()
}

func (l *Residual) LoadFromProvider() {
	l.Layers.LoadFromProvider()
}
