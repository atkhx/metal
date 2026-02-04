package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

type NoopLayer struct{}

func (l *NoopLayer) Compile(device *proc.Device, input *num.Data) *num.Data {
	return input
}

type VAEEncoder struct {
	Layers Layers `json:"VAEEncoder"`
}

func (l *VAEEncoder) Compile(device *proc.Device, input *num.Data) *num.Data {
	_, output := l.Layers.Compile(device, input)
	return output
}

func (l *VAEEncoder) ForUpdate() []*num.Data {
	return l.Layers.ForUpdate()
}

func (l *VAEEncoder) LoadFromProvider() {
	l.Layers.LoadFromProvider()
}

type VAEDecoder struct {
	Layers Layers `json:"VAEDecoder"`
}

func (l *VAEDecoder) Compile(device *proc.Device, input *num.Data) *num.Data {
	_, output := l.Layers.Compile(device, input)
	return output
}

func (l *VAEDecoder) ForUpdate() []*num.Data {
	return l.Layers.ForUpdate()
}

func (l *VAEDecoder) LoadFromProvider() {
	l.Layers.LoadFromProvider()
}
