package num

import "github.com/atkhx/metal/mtl"

type Data struct {
	Data *mtl.Buffer
	Grad *mtl.Buffer
	Dims mtl.MTLSize
	Deps []*Data

	CalcData func(b *mtl.CommandBuffer)
	CalcGrad func(b *mtl.CommandBuffer)

	SkipResetGrad bool
}
