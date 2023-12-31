package mps

import (
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestMatrixRandomMTGP32(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	buffer := device.NewBufferWithFloats([]float32{0, 0, 0}, mtl.ResourceStorageModeShared)
	defer buffer.Release()

	matrixDescriptor := CreateMatrixDescriptorFloat32(3, 1, 1, 3)
	defer matrixDescriptor.Release()

	matrix := CreateMatrixWithBuffer(matrixDescriptor, buffer, 0)
	defer matrix.Release()

	commandQueue := device.NewCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetNewMTLCommandBuffer()
	defer commandBuffer.Release()

	distributionDescriptor := CreateMatrixRandomDistributionDescriptor(0, 1)
	defer distributionDescriptor.Release()

	matrixRandomMTGP32 := CreateMatrixRandomMTGP32(device, distributionDescriptor, 123)
	defer matrixRandomMTGP32.Release()

	matrixRandomMTGP32.Encode(commandBuffer, matrix)
	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()

	require.Equal(t, []float32{0.31826174, 0.5106553, 0.6946397}, matrix.GetData().GetFloats())
}
