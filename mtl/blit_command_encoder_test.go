package mtl

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMTLBlitCommandEncoder_FillBuffer(t *testing.T) {
	device, err := CreateSystemDefaultDevice()
	require.NoError(t, err)
	defer device.Release()

	commandQueue := device.NewCommandQueue()
	defer commandQueue.Release()

	commandBuffer := commandQueue.GetNewMTLCommandBuffer()
	defer commandBuffer.Release()

	buffer := device.NewBufferWithBytes([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9}, ResourceStorageModeShared)
	defer buffer.Release()

	encoder := commandBuffer.GetMTLBlitCommandEncoder()
	encoder.FillBuffer(buffer, NSRange{2, 4}, 0)
	encoder.FillBuffer(buffer, NSRange{0, 3}, 1)
	encoder.EndEncoding()
	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()

	require.Equal(t, []byte{1, 1, 1, 0, 0, 0, 7, 8, 9}, buffer.GetBytes())
}
