package mps

import (
	"math"
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestMatrixSoftMaxKernel(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	mInf := float32(math.Inf(-1))
	data := []float32{
		0.7, mInf, mInf,
		0.01, 0.5, mInf,
		-1, 2, 3,
	}

	// Input matrix
	inputBuffer := device.NewBufferWithFloats(data, mtl.ResourceStorageModeShared)
	defer inputBuffer.Release()

	inputDescriptor := CreateMatrixDescriptorFloat32(3, 3, 1, 3*3)
	defer inputDescriptor.Release()

	inputMatrix := CreateMatrixWithBuffer(inputDescriptor, inputBuffer, 0)
	defer inputMatrix.Release()

	// Input Grad matrix
	inputGradBuffer := device.NewBufferWithFloats([]float32{0, 0, 0, 0, 0, 0, 0, 0, 0}, mtl.ResourceStorageModeShared)
	defer inputGradBuffer.Release()

	inputGradDescriptor := CreateMatrixDescriptorFloat32(3, 3, 1, 3*3)
	defer inputGradDescriptor.Release()

	inputGradMatrix := CreateMatrixWithBuffer(inputGradDescriptor, inputGradBuffer, 0)
	defer inputGradMatrix.Release()

	outputBuffer := device.NewBufferWithFloats([]float32{0, 0, 0, 0, 0, 0, 0, 0, 0}, mtl.ResourceStorageModeShared)
	defer outputBuffer.Release()

	outputDescriptor := CreateMatrixDescriptorFloat32(3, 3, 1, 3*3)
	defer outputDescriptor.Release()

	outputMatrix := CreateMatrixWithBuffer(outputDescriptor, outputBuffer, 0)
	defer outputMatrix.Release()

	// Output Grad matrix
	outputGradBuffer := device.NewBufferWithFloats([]float32{
		0.1, 0, 0,
		0.1, 0.1, 0,
		0.1, 0.1, 0.1,
	}, mtl.ResourceStorageModeShared)
	defer outputGradBuffer.Release()

	outputGradDescriptor := CreateMatrixDescriptorFloat32(3, 3, 1, 3*3)
	defer outputGradDescriptor.Release()

	outputGradMatrix := CreateMatrixWithBuffer(outputGradDescriptor, outputGradBuffer, 0)
	defer outputGradMatrix.Release()

	commandQueue := device.NewCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetNewMTLCommandBuffer()
	defer commandBuffer.Release()

	softmaxKernel := CreateMatrixSoftMaxKernel(device)
	defer softmaxKernel.Release()

	softmaxGradientKernel := MatrixSoftMaxGradientKernelCreate(device)
	defer softmaxGradientKernel.Release()

	softmaxKernel.Encode(commandBuffer, inputMatrix, outputMatrix)
	softmaxGradientKernel.Encode(commandBuffer, outputGradMatrix, outputMatrix, inputGradMatrix)

	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()

	require.Equal(t, []float32{
		1, 0, 0,
		0.37989357, 0.62010646, 0,
		0.0132128885, 0.26538795, 0.72139925,
	}, outputMatrix.GetData().GetFloats())

	t.Log(inputGradMatrix.GetData().GetFloats())
}
