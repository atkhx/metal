package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewDropout(prob float32) *Dropout {
	return &Dropout{dropoutProb: prob}
}

type Dropout struct {
	dropoutProb float32
}

func (l *Dropout) Compile(device *proc.Device, input *num.Data) *num.Data {
	return device.Dropout(input, l.dropoutProb)
}
