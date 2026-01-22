package layer

import (
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

type Layers []Layer

func (s Layers) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	for _, layer := range s {
		inputs = layer.Compile(device, inputs)
	}
	return inputs
}

func (s Layers) CompileAndReturnDataObjects(device *proc.Device, input *num.Data) ([]*num.Data, *num.Data) {
	var output *num.Data
	result := make([]*num.Data, 0, len(s))
	inputs := input
	for _, layer := range s {
		output = layer.Compile(device, inputs)
		result = append(result, output)
		inputs = output
	}
	return result, output
}

func (s Layers) ForUpdate() []*num.Data {
	result := make([]*num.Data, 0, len(s))
	for _, layer := range s {
		if l, ok := layer.(Updatable); ok {
			result = append(result, l.ForUpdate()...)
		}
	}
	return result
}

func (s Layers) LoadFromProvider() {
	for _, ll := range s {
		if l, ok := ll.(WithWeightsProvider); ok {
			l.LoadFromProvider()
		}
	}
}
