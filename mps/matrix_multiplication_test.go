package mps

import (
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestMatrixMultiplicationKernel(t *testing.T) {
	device, err := mtl.CreateSystemDefaultDevice()
	require.NoError(t, err)
	defer device.Release()

	dataA := []float32{
		1, 2, 3,
		4, 5, 6,
	}

	dataB := []float32{
		1, 0,
		0, 1,
		1, 0,
	}

	dataC := []float32{
		0, 0,
		0, 0,
	}

	expected := []float32{
		4, 2,
		10, 5,
	}

	bufferA := device.NewBufferWithFloats(dataA, mtl.ResourceStorageModeShared)
	defer bufferA.Release()

	bufferB := device.NewBufferWithFloats(dataB, mtl.ResourceStorageModeShared)
	defer bufferB.Release()

	bufferC := device.NewBufferWithFloats(dataC, mtl.ResourceStorageModeShared)
	defer bufferC.Release()

	descriptorA := CreateMatrixDescriptorFloat32(3, 2, 1, 3*2)
	descriptorB := CreateMatrixDescriptorFloat32(2, 3, 1, 2*3)
	descriptorC := CreateMatrixDescriptorFloat32(2, 2, 1, 2*2)

	defer descriptorA.Release()
	defer descriptorB.Release()
	defer descriptorC.Release()

	matrixA := CreateMatrixWithBuffer(descriptorA, bufferA, 0)
	matrixB := CreateMatrixWithBuffer(descriptorB, bufferB, 0)
	matrixC := CreateMatrixWithBuffer(descriptorC, bufferC, 0)

	defer matrixA.Release()
	defer matrixB.Release()
	defer matrixC.Release()

	commandQueue := device.NewCommandQueue()
	defer commandQueue.Release()

	commandBuffer, err := CommandBufferFromCommandQueue(commandQueue)
	require.NoError(t, err)
	defer commandBuffer.Release()

	rootCommandBuffer := commandBuffer.GetRootMTLCommandBuffer()

	kernel := CreateMatrixMultiplicationKernel(device, 2, 2, 3, 1.0, 0, false, false)
	kernel.Encode(commandBuffer.GetMTLCommandBuffer(), matrixA, matrixB, matrixC)
	defer kernel.Release()

	commandBuffer.CommitAndContinue()
	rootCommandBuffer.WaitUntilCompleted()

	require.Equal(t, expected, bufferC.GetFloats())
}
