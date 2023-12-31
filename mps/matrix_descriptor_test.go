package mps

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

func TestMPSMatrixDescriptorFloat32(t *testing.T) {
	sizeOfFloat32 := int(unsafe.Sizeof(float32(0)))

	cols, rows := 5, 3
	batchSize := 2
	batchStride := cols * rows

	desc := CreateMatrixDescriptorFloat32(cols, rows, batchSize, batchStride)
	defer desc.Release()

	require.Equal(t, rows, desc.GetRows())
	require.Equal(t, cols, desc.GetColumns())
	require.Equal(t, batchSize, desc.GetMatrices())
	t.Log(desc.GetDataType())
	require.Equal(t, cols*sizeOfFloat32, desc.GetRowBytes())
	require.Equal(t, batchStride*sizeOfFloat32, desc.GetMatrixBytes())
}
