package layer

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func initWeights(device *proc.Device, init initializer.Initializer, dims mtl.MTLSize, fanIn, fanOut int) *num.Data {
	w := init.GetNormK(fanIn, fanOut)
	if dist, ok := init.(initializer.DistributionProvider); ok && dist.Distribution() == initializer.DistributionNormal {
		return device.NewDataRandNormalWeighted(dims, w)
	}
	return device.NewDataRandUniformWeighted(dims, w)
}
