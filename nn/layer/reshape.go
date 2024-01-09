package layer

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewReshape(dims mtl.MTLSize) *Reshape {
	return &Reshape{dims: dims}
}

type Reshape struct {
	dims mtl.MTLSize
}

func (l *Reshape) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	return device.Reshape(inputs, l.dims)
}
