package dropout

import (
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/stretchr/testify/require"
)

func TestDropout_InvertedScaling(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	inputBuf := device.NewBufferWithFloats([]float32{1, 2, 3, 4}, mtl.ResourceStorageModeShared)
	inputGrad := device.NewBufferEmptyFloatsBuffer(4, mtl.ResourceStorageModeShared)
	outputBuf := device.NewBufferEmptyFloatsBuffer(4, mtl.ResourceStorageModeShared)
	outputGrad := device.NewBufferWithFloats([]float32{1, 1, 1, 1}, mtl.ResourceStorageModeShared)

	input := &num.Data{
		Data: inputBuf,
		Grad: inputGrad,
		Dims: mtl.NewMTLSize(4),
	}
	output := &num.Data{
		Data: outputBuf,
		Grad: outputGrad,
		Dims: mtl.NewMTLSize(4),
	}

	kernel := New(device, input, output, 0.5, 1)
	scale := float32(2.0)

	cmd := device.NewCommandQueue().GetNewMTLCommandBuffer()
	defer cmd.Release()
	kernel.Forward(cmd)
	kernel.Backward(cmd)
	cmd.Commit()
	cmd.WaitUntilCompleted()

	for i, inVal := range input.Data.GetFloats() {
		outVal := output.Data.GetFloats()[i]
		if outVal != 0 {
			require.InDelta(t, float64(inVal*scale), float64(outVal), 1e-5)
			require.InDelta(t, float64(scale), float64(input.Grad.GetFloats()[i]), 1e-5)
		} else {
			require.InDelta(t, 0.0, float64(input.Grad.GetFloats()[i]), 1e-5)
		}
	}
}
