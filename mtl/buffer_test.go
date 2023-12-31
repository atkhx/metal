package mtl

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewBufferWithBytes(t *testing.T) {
	device := MustCreateSystemDefaultDevice()
	require.NotNil(t, device.GetID())
	defer device.Release()

	buffer := device.NewBufferWithBytes([]byte{1, 2, 3}, ResourceStorageModeShared)
	defer buffer.Release()

	require.Equal(t, uint64(3), buffer.GetLengthBytes())

	contents := buffer.GetBytes()
	require.Equal(t, []byte{1, 2, 3}, contents)
}

func TestNewBufferWithFloats(t *testing.T) {
	device := MustCreateSystemDefaultDevice()
	require.NotNil(t, device.GetID())
	defer device.Release()

	buffer := device.NewBufferWithFloats([]float32{1, 2, 3}, ResourceStorageModeShared)
	defer buffer.Release()

	require.Equal(t, uint64(3), buffer.GetLengthFloats())

	contents := buffer.GetFloats()
	require.Equal(t, []float32{1, 2, 3}, contents)
}
