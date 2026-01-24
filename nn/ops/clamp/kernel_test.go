package clamp

import (
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/stretchr/testify/require"
)

func TestClamp_ForwardBackward(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	inputBuf := device.NewBufferWithFloats([]float32{-2, -1, 0, 1, 2}, mtl.ResourceStorageModeShared)
	outputBuf := device.NewBufferEmptyFloatsBuffer(5, mtl.ResourceStorageModeShared)
	outputGradBuf := device.NewBufferWithFloats([]float32{1, 1, 1, 1, 1}, mtl.ResourceStorageModeShared)
	inputGradBuf := device.NewBufferEmptyFloatsBuffer(5, mtl.ResourceStorageModeShared)

	input := &num.Data{Data: inputBuf, Grad: inputGradBuf, Dims: mtl.NewMTLSize(5)}
	output := &num.Data{Data: outputBuf, Grad: outputGradBuf, Dims: mtl.NewMTLSize(5)}

	kernel := New(device, input, output, -1, 1)

	b := device.NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	kernel.Forward(b)
	kernel.Backward(b)
	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, []float32{-1, -1, 0, 1, 1}, output.Data.GetFloats())
	require.Equal(t, []float32{0, 1, 1, 1, 0}, input.Grad.GetFloats())
}
