package sanitize

import (
	"math"
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/stretchr/testify/require"
)

func TestSanitize_ForwardBackward(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	inputBuf := device.NewBufferWithFloats([]float32{0, float32(math.NaN()), 2, float32(math.Inf(1))}, mtl.ResourceStorageModeShared)
	outputBuf := device.NewBufferEmptyFloatsBuffer(4, mtl.ResourceStorageModeShared)
	outputGradBuf := device.NewBufferWithFloats([]float32{1, 1, 1, 1}, mtl.ResourceStorageModeShared)
	inputGradBuf := device.NewBufferEmptyFloatsBuffer(4, mtl.ResourceStorageModeShared)

	input := &num.Data{Data: inputBuf, Grad: inputGradBuf, Dims: mtl.NewMTLSize(4)}
	output := &num.Data{Data: outputBuf, Grad: outputGradBuf, Dims: mtl.NewMTLSize(4)}

	kernel := New(device, input, output)

	b := device.NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	kernel.Forward(b)
	kernel.Backward(b)
	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, []float32{0, 0, 2, 0}, output.Data.GetFloats())
	require.Equal(t, []float32{1, 0, 1, 0}, input.Grad.GetFloats())
}
