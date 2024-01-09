package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

type Layer interface {
	Compile(device *proc.Device, inputs *num.Data) *num.Data
}

type Updatable interface {
	ForUpdate() []*num.Data
}

type WithWeightsProvider interface {
	LoadFromProvider()
}
