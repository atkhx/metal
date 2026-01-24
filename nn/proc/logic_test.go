package proc

import (
	"math"
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestDevice_MeanGrad(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	input := device.NewDataWithValues(mtl.NewMTLSize(4), []float32{1, 2, 3, 4})
	output := device.Mean(input)

	outputGrad := output.Grad.GetFloats()
	outputGrad[0] = 1

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	output.CalcData(b)
	output.CalcGrad(b)
	b.Commit()
	b.WaitUntilCompleted()

	require.InDelta(t, 2.5, float64(output.Data.GetFloats()[0]), 1e-5)

	for _, v := range input.Grad.GetFloats() {
		require.InDelta(t, 0.25, float64(v), 1e-5)
	}
}

func TestDevice_SiLuGrad(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	inputVals := []float32{-1, 0, 1}
	input := device.NewDataWithValues(mtl.NewMTLSize(3), inputVals)
	output := device.SiLu(input)

	for i := range output.Grad.GetFloats() {
		output.Grad.GetFloats()[i] = 1
	}

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	output.CalcData(b)
	output.CalcGrad(b)
	b.Commit()
	b.WaitUntilCompleted()

	for i, v := range inputVals {
		sigmoid := 1.0 / (1.0 + math.Exp(-float64(v)))
		expected := sigmoid + float64(v)*sigmoid*(1.0-sigmoid)
		require.InDelta(t, expected, float64(input.Grad.GetFloats()[i]), 1e-4)
	}
}

func TestDevice_ConvStrideForward(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	input := device.NewDataWithValues(mtl.NewMTLSize(4, 4, 1), []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	})

	weights := device.NewDataWithValues(mtl.NewMTLSize(2, 2, 1), []float32{
		1, 1,
		1, 1,
	})
	biases := device.NewDataWithValues(mtl.NewMTLSize(1, 1, 1), []float32{0})

	output := device.Conv(input, weights, biases, 1, 1, 0, 2)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	output.CalcData(b)
	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, []float32{
		14, 22,
		46, 54,
	}, output.Data.GetFloats())
}

func TestDevice_ConvInputGradAccumulate(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	input := device.NewDataWithValues(mtl.NewMTLSize(2, 2, 1), []float32{
		1, 2,
		3, 4,
	})
	weights := device.NewDataWithValues(mtl.NewMTLSize(1, 1, 1), []float32{2})
	biases := device.NewDataWithValues(mtl.NewMTLSize(1, 1, 1), []float32{0})

	output := device.Conv(input, weights, biases, 1, 1, 0, 1)

	copy(output.Grad.GetFloats(), []float32{
		1, 2,
		3, 4,
	})
	for i := range input.Grad.GetFloats() {
		input.Grad.GetFloats()[i] = 1
	}

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	output.CalcGrad(b)
	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, []float32{
		1 + 2*1, 1 + 2*2,
		1 + 2*3, 1 + 2*4,
	}, input.Grad.GetFloats())
}

func TestDevice_TrilMaskAccumulation(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	input := device.NewDataWithValues(mtl.NewMTLSize(2, 2, 1), []float32{
		1, 2,
		3, 4,
	})
	output := device.TrilMask(input)

	copy(output.Grad.GetFloats(), []float32{
		1, 2,
		3, 4,
	})
	for i := range input.Grad.GetFloats() {
		input.Grad.GetFloats()[i] = 1
	}

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	output.CalcGrad(b)
	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, []float32{
		2, 1,
		4, 5,
	}, input.Grad.GetFloats())
}
