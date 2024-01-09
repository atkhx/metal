package softmax

import (
	"github.com/atkhx/metal/mps"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

func New(device *mtl.Device, input, output *num.Data) *Kernel {
	descriptor := mps.CreateMatrixDescriptorFloat32(input.Dims.W, input.Dims.H*input.Dims.D, 1, input.Dims.H*input.Dims.D)

	return &Kernel{
		forwardKernel:  mps.CreateMatrixSoftMaxKernel(device),
		backwardKernel: mps.CreateMatrixSoftMaxGradientKernel(device),

		inputDataMatrix:  mps.CreateMatrixWithBuffer(descriptor, input.Data, 0),
		inputGradMatrix:  mps.CreateMatrixWithBuffer(descriptor, input.Grad, 0),
		outputDataMatrix: mps.CreateMatrixWithBuffer(descriptor, output.Data, 0),
		outputGradMatrix: mps.CreateMatrixWithBuffer(descriptor, output.Grad, 0),
	}
}

type Kernel struct {
	forwardKernel  *mps.MatrixSoftMaxKernel
	backwardKernel *mps.MatrixSoftMaxGradientKernel

	inputDataMatrix  *mps.Matrix
	inputGradMatrix  *mps.Matrix
	outputDataMatrix *mps.Matrix
	outputGradMatrix *mps.Matrix
}

func (op *Kernel) Forward(b *mtl.CommandBuffer) {
	op.forwardKernel.Encode(b, op.inputDataMatrix, op.outputDataMatrix)
}

func (op *Kernel) Backward(b *mtl.CommandBuffer) {
	op.backwardKernel.Encode(b, op.outputGradMatrix, op.outputDataMatrix, op.inputGradMatrix)
}
