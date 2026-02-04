package pkg

import (
	"github.com/atkhx/metal/nn/proc"
)

const (
	adamBeta1        = 0.9
	adamBeta2        = 0.99
	adamLearningRate = 0.0003
	adamEPS          = 0.000000001
)

func CreateOptimizer(epochs int, device *proc.Device) proc.Optimizer {
	return device.GetOptimizerAdam(epochs, adamBeta1, adamBeta2, adamLearningRate, adamEPS)
}
