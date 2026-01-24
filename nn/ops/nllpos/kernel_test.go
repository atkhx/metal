package nllpos

import (
	"math"
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/stretchr/testify/require"
)

func TestNLLPos_ClampsSoftmax(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	softmaxBuf := device.NewBufferWithFloats([]float32{0, 1}, mtl.ResourceStorageModeShared)
	targetsBuf := device.NewBufferWithFloats([]float32{0}, mtl.ResourceStorageModeShared)
	outputBuf := device.NewBufferEmptyFloatsBuffer(1, mtl.ResourceStorageModeShared)

	input := &num.Data{Data: softmaxBuf, Grad: device.NewBufferEmptyFloatsBuffer(2, mtl.ResourceStorageModeShared), Dims: mtl.NewMTLSize(2)}
	targets := &num.Data{Data: targetsBuf, Grad: device.NewBufferEmptyFloatsBuffer(1, mtl.ResourceStorageModeShared), Dims: mtl.NewMTLSize(1)}
	output := &num.Data{Data: outputBuf, Grad: device.NewBufferEmptyFloatsBuffer(1, mtl.ResourceStorageModeShared), Dims: mtl.NewMTLSize(1)}

	kernel := New(device, input, output, targets, 2)

	b := device.NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	kernel.Forward(b)
	b.Commit()
	b.WaitUntilCompleted()

	got := output.Data.GetFloats()[0]
	require.False(t, math.IsInf(float64(got), 0))
	require.False(t, math.IsNaN(float64(got)))

	expected := -math.Log(1e-9)
	require.InDelta(t, expected, float64(got), 1e-3)
}

func TestNLLPos_BackwardFinite(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	softmaxBuf := device.NewBufferWithFloats([]float32{0, 1}, mtl.ResourceStorageModeShared)
	targetsBuf := device.NewBufferWithFloats([]float32{0}, mtl.ResourceStorageModeShared)
	outputBuf := device.NewBufferWithFloats([]float32{0}, mtl.ResourceStorageModeShared)
	outputGradBuf := device.NewBufferWithFloats([]float32{1}, mtl.ResourceStorageModeShared)
	inputGradBuf := device.NewBufferEmptyFloatsBuffer(2, mtl.ResourceStorageModeShared)

	input := &num.Data{Data: softmaxBuf, Grad: inputGradBuf, Dims: mtl.NewMTLSize(2)}
	targets := &num.Data{Data: targetsBuf, Grad: device.NewBufferEmptyFloatsBuffer(1, mtl.ResourceStorageModeShared), Dims: mtl.NewMTLSize(1)}
	output := &num.Data{Data: outputBuf, Grad: outputGradBuf, Dims: mtl.NewMTLSize(1)}

	kernel := New(device, input, output, targets, 2)

	b := device.NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	kernel.Backward(b)
	b.Commit()
	b.WaitUntilCompleted()

	grads := input.Grad.GetFloats()
	for _, g := range grads {
		require.False(t, math.IsInf(float64(g), 0))
		require.False(t, math.IsNaN(float64(g)))
	}
	require.NotEqual(t, float32(0), grads[0])
}
