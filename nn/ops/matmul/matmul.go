package matmul

import (
	"github.com/atkhx/metal/mps"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

type Kernel interface {
	Forward(b *mtl.CommandBuffer)
	Backward(b *mtl.CommandBuffer)
}

func createMatrices3D(aData *num.Data, batchSize, batchStrideK int) (*mps.Matrix, *mps.Matrix) {
	aDesc := mps.CreateMatrixDescriptorFloat32(
		aData.Dims.W,
		aData.Dims.H,
		batchSize,
		aData.Dims.W*aData.Dims.H*batchStrideK,
	)

	return mps.CreateMatrixWithBuffer(aDesc, aData.Data, 0),
		mps.CreateMatrixWithBuffer(aDesc, aData.Grad, 0)
}

func New(device *mtl.Device, aData, bData, cData *num.Data, alpha float32) Kernel {
	if aData.Dims.D == bData.Dims.D {
		return NewEqual(device, aData, bData, cData, alpha)
	}

	if aData.Dims.D == 1 {
		return NewFlat(device, aData, bData, cData, alpha)
	}

	if bData.Dims.D == 1 {
		return NewOnFlat(device, aData, bData, cData, alpha)
	}

	panic("not implemented")
}
