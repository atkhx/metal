package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

type LayersBlock struct {
	Layers Layers
}

func (l *LayersBlock) Compile(device *proc.Device, input *num.Data) *num.Data {
	_, output := l.Layers.Compile(device, input)
	return output
}

func (l *LayersBlock) ForUpdate() []*num.Data {
	return l.Layers.ForUpdate()
}

func (l *LayersBlock) LoadFromProvider() {
	l.Layers.LoadFromProvider()
}
