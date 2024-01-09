package matmul

import (
	"github.com/atkhx/metal/mps"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

func NewFlat(device *mtl.Device, aData, bData, cData *num.Data, alpha float32) *FlatKernel {
	if aData.Dims.D != 1 {
		panic("aDepth != 1")
	}

	op := &FlatKernel{}

	aW, aH, _ := aData.Dims.GetWHD()
	bW, bH, bD := bData.Dims.GetWHD()
	cW, cH, _ := cData.Dims.GetWHD()

	batchSize := bD

	op.aDataM, op.aGradM = createMatrices3D(aData, batchSize, 0)
	op.bDataM, op.bGradM = createMatrices3D(bData, batchSize, 1)
	op.cDataM, op.cGradM = createMatrices3D(cData, batchSize, 1)

	op.calcCData = mps.CreateMatrixMultiplicationKernel(device, aH, bW, aW, alpha, 0.0, false, false)
	op.calcBGrad = mps.CreateMatrixMultiplicationKernel(device, bH, bW, cH, alpha, 1.0, true, false)
	op.calcAGrad = mps.CreateMatrixMultiplicationKernel(device, aH, aW, cW, alpha, 1.0, false, true)

	op.bDataMs = make([]*mps.Matrix, 0, batchSize)
	op.cGradMs = make([]*mps.Matrix, 0, batchSize)

	bDescOne := mps.CreateMatrixDescriptorFloat32(bW, bH, 1, 0)
	cDescOne := mps.CreateMatrixDescriptorFloat32(cW, cH, 1, 0)

	for i := 0; i < batchSize; i++ {
		op.bDataMs = append(op.bDataMs, mps.CreateMatrixWithBuffer(bDescOne, bData.Data, i*bW*bH))
		op.cGradMs = append(op.cGradMs, mps.CreateMatrixWithBuffer(cDescOne, cData.Grad, i*cW*cH))
	}

	return op
}

type FlatKernel struct {
	calcCData *mps.MatrixMultiplicationKernel
	calcAGrad *mps.MatrixMultiplicationKernel
	calcBGrad *mps.MatrixMultiplicationKernel

	aDataM, bDataM, cDataM *mps.Matrix
	aGradM, bGradM, cGradM *mps.Matrix

	cGradMs, bDataMs []*mps.Matrix
}

func (op *FlatKernel) Forward(b *mtl.CommandBuffer) {
	op.calcCData.Encode(b, op.aDataM, op.bDataM, op.cDataM)
}

func (op *FlatKernel) Backward(b *mtl.CommandBuffer) {
	op.calcBGrad.Encode(b, op.aDataM, op.cGradM, op.bGradM)
	for i := 0; i < len(op.cGradMs); i++ {
		i := i
		op.calcAGrad.Encode(b, op.cGradMs[i], op.bDataMs[i], op.aGradM)
	}
}
