package matmul

import (
	"github.com/atkhx/metal/mps"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

func NewOnFlat(device *mtl.Device, aData, bData, cData *num.Data, alpha float32) *OnFlatKernel {
	if bData.Dims.D != 1 {
		panic("bDepth != 1")
	}

	op := &OnFlatKernel{}

	aW, aH, aD := aData.Dims.GetWHD()
	bW, bH, _ := bData.Dims.GetWHD()
	cW, cH, cD := cData.Dims.GetWHD()

	batchSize := aD

	op.aDataM, op.aGradM = createMatrices3D(aData, batchSize, 1)
	op.bDataM, op.bGradM = createMatrices3D(bData, batchSize, 0)
	op.cDataM, op.cGradM = createMatrices3D(cData, batchSize, 1)

	aDescOne := mps.CreateMatrixDescriptorFloat32(aW, aH*aD, 1, aData.Dims.Length())
	cDescOne := mps.CreateMatrixDescriptorFloat32(cW, cH*cD, 1, cData.Dims.Length())

	op.aDataMBig = mps.CreateMatrixWithBuffer(aDescOne, aData.Data, 0)
	op.cGradMBig = mps.CreateMatrixWithBuffer(cDescOne, cData.Grad, 0)

	op.calcCData = mps.CreateMatrixMultiplicationKernel(device, aH, bW, aW, alpha, 0.0, false, false)
	op.calcAGrad = mps.CreateMatrixMultiplicationKernel(device, aH, aW, cW, alpha, 1.0, false, true)
	op.calcBGrad = mps.CreateMatrixMultiplicationKernel(device, bH, bW, aH*aD, alpha, 1.0, true, false)

	return op
}

type OnFlatKernel struct {
	calcCData *mps.MatrixMultiplicationKernel
	calcAGrad *mps.MatrixMultiplicationKernel
	calcBGrad *mps.MatrixMultiplicationKernel

	aDataM, bDataM, cDataM *mps.Matrix
	aGradM, bGradM, cGradM *mps.Matrix

	aDataMBig, cGradMBig *mps.Matrix
}

func (op *OnFlatKernel) Forward(b *mtl.CommandBuffer) {
	op.calcCData.Encode(b, op.aDataM, op.bDataM, op.cDataM)
}

func (op *OnFlatKernel) Backward(b *mtl.CommandBuffer) {
	op.calcAGrad.Encode(b, op.cGradM, op.bDataM, op.aGradM)
	op.calcBGrad.Encode(b, op.aDataMBig, op.cGradMBig, op.bGradM)
}
