package mps

import (
	"testing"
	"unsafe"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestMPSMatrix(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	rows, cols := 3, 2
	batchSize := 2
	batchStride := cols * rows
	sizeOfFloat := int(unsafe.Sizeof(float32(0)))
	offset := 3
	offsetInBytes := offset * sizeOfFloat

	buffer := device.NewBufferWithFloats([]float32{
		0, 0, 0,

		1, 2, 3,
		4, 5, 6,

		3, 4, 5,
		6, 7, 8,
	}, mtl.ResourceStorageModeShared)

	descriptor := CreateMatrixDescriptorFloat32(cols, rows, batchSize, batchStride)
	defer descriptor.Release()

	matrix := CreateMatrixWithBuffer(descriptor, buffer, offsetInBytes)
	defer matrix.Release()

	require.Equal(t, rows, matrix.GetRows())
	require.Equal(t, cols, matrix.GetColumns())
	require.Equal(t, batchSize, matrix.GetMatrices())
	t.Log(matrix.GetDataType())
	require.Equal(t, cols*sizeOfFloat, matrix.GetRowBytes())
	require.Equal(t, batchStride*sizeOfFloat, matrix.GetMatrixBytes())
	require.Equal(t, offsetInBytes, matrix.GetOffset())

	data := matrix.GetData()

	require.Equal(t, []float32{
		1, 2, 3,
		4, 5, 6,

		3, 4, 5,
		6, 7, 8,
	}, data.GetFloats()[offset:])

	t.Log(matrix.GetResourceSize())
}
