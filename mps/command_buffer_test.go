package mps

import (
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestMPSCommandBufferFromCommandQueue(t *testing.T) {
	device, err := mtl.CreateSystemDefaultDevice()
	require.NoError(t, err)
	defer device.Release()

	commandQueue := device.NewCommandQueue()
	defer commandQueue.Release()

	commandBuffer, err := CommandBufferFromCommandQueue(commandQueue)
	require.NoError(t, err)
	defer commandBuffer.Release()

	mtlCommandBufferID := commandBuffer.GetID()
	require.NotNil(t, mtlCommandBufferID)

	rootMTLCommandBufferID := commandBuffer.GetRootMTLCommandBufferID()
	require.NotNil(t, rootMTLCommandBufferID)
}

func TestMPSCommandBufferWithBlitEncoder(t *testing.T) {
	device, err := mtl.CreateSystemDefaultDevice()
	require.NoError(t, err)
	defer device.Release()

	commandQueue := device.NewCommandQueue()
	defer commandQueue.Release()

	mpsCommandBuffer, err := CommandBufferFromCommandQueue(commandQueue)
	require.NoError(t, err)
	defer mpsCommandBuffer.Release()

	buffer := device.NewBufferWithBytes([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}, mtl.ResourceStorageModeShared)
	defer buffer.Release()

	commandBuffer1 := mpsCommandBuffer.GetMTLCommandBuffer()
	encoder := commandBuffer1.GetMTLBlitCommandEncoder()
	encoder.FillBuffer(buffer, mtl.NSRange{Location: 2, Length: 4}, 0)
	encoder.FillBuffer(buffer, mtl.NSRange{Location: 0, Length: 3}, 1)
	encoder.EndEncoding()
	mpsCommandBuffer.CommitAndContinue()

	commandBuffer2 := mpsCommandBuffer.GetMTLCommandBuffer()
	encoder = commandBuffer2.GetMTLBlitCommandEncoder()
	encoder.FillBuffer(buffer, mtl.NSRange{Location: 4, Length: 2}, 2)
	encoder.EndEncoding()
	commandBuffer2.Commit() // The last command buffer should be committed directly.

	rootCommandBuffer := mpsCommandBuffer.GetRootMTLCommandBuffer()
	rootCommandBuffer.WaitUntilCompleted()
	require.Equal(t, []byte{1, 1, 1, 0, 2, 2, 7, 8, 9}, buffer.GetBytes())
}
