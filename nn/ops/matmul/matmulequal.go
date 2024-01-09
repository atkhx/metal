package matmul

import (
	"github.com/atkhx/metal/mps"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

func NewEqual(device *mtl.Device, aData, bData, cData *num.Data, alpha float32) *Equal {
	if aData.Dims.D != bData.Dims.D {
		panic("aDepth != bDepth")
	}

	op := &Equal{}

	aW, aH, _ := aData.Dims.GetWHD()
	bW, bH, _ := bData.Dims.GetWHD()
	cW, _, _ := cData.Dims.GetWHD()

	op.aDataM, op.aGradM = createMatrices3D(aData, aData.Dims.D, 1)
	op.bDataM, op.bGradM = createMatrices3D(bData, aData.Dims.D, 1)
	op.cDataM, op.cGradM = createMatrices3D(cData, aData.Dims.D, 1)

	op.calcCData = mps.CreateMatrixMultiplicationKernel(device, aH, bW, aW, alpha, 0.0, false, false)
	op.calcAGrad = mps.CreateMatrixMultiplicationKernel(device, aH, aW, cW, alpha, 1.0, false, true)
	op.calcBGrad = mps.CreateMatrixMultiplicationKernel(device, bH, bW, aH, alpha, 1.0, true, false)

	return op
}

type Equal struct {
	calcCData *mps.MatrixMultiplicationKernel
	calcAGrad *mps.MatrixMultiplicationKernel
	calcBGrad *mps.MatrixMultiplicationKernel

	aDataM, bDataM, cDataM *mps.Matrix
	aGradM, bGradM, cGradM *mps.Matrix
}

func (op *Equal) Forward(b *mtl.CommandBuffer) {
	op.calcCData.Encode(b, op.aDataM, op.bDataM, op.cDataM)
}

func (op *Equal) Backward(b *mtl.CommandBuffer) {
	op.calcAGrad.Encode(b, op.cGradM, op.bDataM, op.aGradM)
	op.calcBGrad.Encode(b, op.aDataM, op.cGradM, op.bGradM)
}
