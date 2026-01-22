package proc

import (
	"math"
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestDevice_AddEqual(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(6), []float32{1, 2, 3, 4, 5, 6})
	bData := device.NewDataWithValues(mtl.NewMTLSize(6), []float32{2, 3, 4, 5, 6, 7})
	cData := device.AddEqual(aData, bData)

	expectedResult := []float32{3, 5, 7, 9, 11, 13}
	cGradsValues := []float32{3, 4, 5, 6, 7, 8}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, cGradsValues, aData.Grad.GetFloats())
	require.Equal(t, cGradsValues, bData.Grad.GetFloats())
}

func TestDevice_AddRow(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 2, 2), []float32{
		1, 2, 3,
		4, 5, 6,

		2, 3, 4,
		5, 6, 7,
	})

	bData := device.NewDataWithValues(mtl.NewMTLSize(3), []float32{
		1, 3, 5,
	})

	cData := device.AddRow(aData, bData, 3)

	expectedResult := []float32{
		2, 5, 8,
		5, 8, 11,

		3, 6, 9,
		6, 9, 12,
	}

	cGradsValues := []float32{
		2, 4, 6,
		8, 10, 12,

		1, 3, 5,
		7, 9, 11,
	}

	expectedAGrads := cGradsValues
	expectedBGrads := []float32{
		18, 26, 34,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
	require.Equal(t, expectedBGrads, bData.Grad.GetFloats())
}

func TestDevice_MulCol(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 2, 2), []float32{
		1, 2, 3,
		4, 5, 6,

		2, 3, 4,
		5, 6, 7,
	})

	bData := device.NewDataWithValues(mtl.NewMTLSize(1, 2), []float32{
		2,
		2,
	})

	cData := device.MulCol(aData, bData, 1, 2)

	expectedResult := []float32{
		2, 4, 6,
		8, 10, 12,

		4, 6, 8,
		10, 12, 14,
	}

	cGradsValues := []float32{
		1, 2, 3,
		4, 5, 6,

		3, 4, 5,
		6, 7, 8,
	}

	expectedAGrads := []float32{
		2, 4, 6,
		8, 10, 12,

		6, 8, 10,
		12, 14, 16,
	}

	expectedBGrads := []float32{
		103,
		154,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
	require.Equal(t, expectedBGrads, bData.Grad.GetFloats())
}

func TestDevice_MulRow(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 2, 2), []float32{
		1, 2, 3,
		4, 5, 6,

		2, 3, 4,
		5, 6, 7,
	})

	bData := device.NewDataWithValues(mtl.NewMTLSize(3), []float32{
		2, 2, 2,
	})

	cData := device.MulRow(aData, bData, 3)

	expectedResult := []float32{
		2, 4, 6,
		8, 10, 12,

		4, 6, 8,
		10, 12, 14,
	}

	cGradsValues := []float32{
		1, 2, 3,
		4, 5, 6,

		3, 4, 5,
		6, 7, 8,
	}

	expectedAGrads := []float32{
		2, 4, 6,
		8, 10, 12,

		6, 8, 10,
		12, 14, 16,
	}

	expectedBGrads := []float32{
		53, 83, 121,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
	require.Equal(t, expectedBGrads, bData.Grad.GetFloats())
}

func TestDevice_MulEqual(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 2, 2), []float32{
		1, 2, 3,
		4, 5, 6,

		2, 3, 4,
		5, 6, 7,
	})

	bData := device.NewDataWithValues(mtl.NewMTLSize(3, 2, 2), []float32{
		2, 3, 4,
		5, 6, 7,

		3, 4, 5,
		6, 7, 8,
	})

	cData := device.MulEqual(aData, bData)

	expectedResult := []float32{
		2, 6, 12,
		20, 30, 42,

		6, 12, 20,
		30, 42, 56,
	}

	cGradsValues := []float32{
		1, 2, 3,
		4, 5, 6,

		3, 4, 5,
		6, 7, 8,
	}

	expectedAGrads := []float32{
		2, 6, 12,
		20, 30, 42,

		9, 16, 25,
		36, 49, 64,
	}

	expectedBGrads := []float32{
		1, 4, 9,
		16, 25, 36,

		6, 12, 20,
		30, 42, 56,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
	require.Equal(t, expectedBGrads, bData.Grad.GetFloats())
}

func TestDevice_RMSNorm(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 2), []float32{
		1, 2, 3,
		4, 5, 6,
	})

	cData := device.RMSNorm(aData, 3)

	expectedResult := []float32{
		0.46290955, 0.9258191, 1.3887286,
		0.78954184, 0.9869273, 1.1843128,
	}

	cGradsValues := []float32{
		11, 12, 13,
		14, 15, 16,
	}

	expectedAGrads := []float32{
		2.6452026, 0.66130924, -1.3225837,
		0.4357872, 0.051270187, -0.3332466,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
}

func TestDevice_RopeCols(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	headsCount, headSize, contextLength := 3, 2, 3
	featuresCount := headsCount * headSize

	aData := device.NewDataWithValues(mtl.NewMTLSize(contextLength, headSize, headsCount), []float32{
		1, 2, 3,
		4, 5, 6,

		3, 4, 5,
		6, 7, 8,

		5, 6, 7,
		8, 9, 10,
	})

	cData := device.RopeCols(aData, featuresCount, headSize, contextLength)

	expectedResult := []float32{

		1, -3.12675, -6.704225,
		4, 4.384453, 0.23101056,

		3, -3.7290876, -9.355114,
		6, 7.1479993, 1.2173114,

		5, -4.3314247, -12.006002,
		8, 9.911545, 2.2036123,
	}

	cGradsValues := []float32{
		11, 12, 13,
		14, 15, 16,

		17, 18, 19,
		20, 21, 22,

		23, 24, 25,
		26, 27, 28,
	}

	expectedAGrads := []float32{
		11, 19.10569, 9.138848,
		14, -1.9931173, -18.479218,

		17, 27.396328, 12.097752,
		20, -3.8001292, -26.431885,

		23, 35.68697, 15.056654,
		26, -5.607141, -34.38455,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
}

func TestDevice_Relu(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 3), []float32{
		-1, 0, 1,
		2, 0.1, -1,
		-0, 1, 1,
	})

	cData := device.Relu(aData)

	expectedResult := []float32{
		0, 0, 1,
		2, 0.1, 0,
		0, 1, 1,
	}

	cGradsValues := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}

	expectedAGrads := []float32{
		0, 0, 3,
		4, 5, 0,
		0, 8, 9,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
}

func TestDevice_SiLu(t *testing.T) {
	t.SkipNow()
}

func TestDevice_Dropout(t *testing.T) {
	t.SkipNow()
}

func TestDevice_Reshape(t *testing.T) {
	t.SkipNow()
}

func TestDevice_TrilMask(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 3, 2), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		2, 3, 4,
		5, 6, 7,
		8, 9, 0,
	})

	cData := device.TrilMask(aData)

	i := float32(math.Inf(-1))

	expectedResult := []float32{
		1, i, i,
		4, 5, i,
		7, 8, 9,

		2, i, i,
		5, 6, i,
		8, 9, 0,
	}

	cGradsValues := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		9, 8, 7,
		6, 5, 4,
		3, 2, 1,
	}

	expectedAGrads := []float32{
		1, 0, 0,
		4, 5, 0,
		7, 8, 9,

		9, 0, 0,
		6, 5, 0,
		3, 2, 1,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
}

func TestDevice_Softmax(t *testing.T) {
	t.SkipNow()
}

func TestDevice_TriangleLowerSoftmax(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 3, 2), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		2, 3, 4,
		5, 6, 7,
		8, 9, 1,
	})

	cData := device.TriangleLowerSoftmax(aData)

	pipe := device.GetTestingPipeline(cData)

	expectedResult := []float32{
		1, 0, 0,
		0.2689414, 0.7310586, 0,
		0.09003057, 0.24472848, 0.66524094,

		1, 0, 0,
		0.2689414, 0.7310586, 0,
		0.26887548, 0.73087937, 0.00024518272,
	}

	cGradsValues := []float32{
		1, 2, 3,
		1, 2, 3,
		1, 2, 3,

		1, 2, 3,
		1, 2, 3,
		1, 2, 3,
	}

	expectedAGrads := []float32{
		0, 0, 0,
		-0.19661193, 0.19661193, 0,
		-0.1418171, -0.14077035, 0.28258747,

		0, 0, 0,
		-0.19661193, 0.19661193, 0,
		-0.19664739, 0.19633631, 0.00031104623,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	pipe.Forward()
	pipe.Backward()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
}

func TestDevice_Transpose(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	aData := device.NewDataWithValues(mtl.NewMTLSize(3, 3, 2), []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		2, 3, 4,
		5, 6, 7,
		8, 9, 0,
	})

	cData := device.Transpose(aData)

	expectedResult := []float32{
		1, 4, 7,
		2, 5, 8,
		3, 6, 9,

		2, 5, 8,
		3, 6, 9,
		4, 7, 0,
	}

	cGradsValues := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,

		2, 3, 4,
		5, 6, 7,
		8, 9, 0,
	}

	expectedAGrads := []float32{
		1, 4, 7,
		2, 5, 8,
		3, 6, 9,

		2, 5, 8,
		3, 6, 9,
		4, 7, 0,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedAGrads, aData.Grad.GetFloats())
}

func TestDevice_Embeddings(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	tEmbeddings := device.NewDataWithValues(mtl.NewMTLSize(3, 6), []float32{
		1, 2, 3,
		2, 3, 4,
		3, 4, 5,

		4, 5, 6,
		5, 6, 7,
		6, 7, 8,
	})

	aData := device.NewDataWithValues(mtl.NewMTLSize(3), []float32{
		0, 3, 5,
	})

	cData := device.Embeddings(aData, tEmbeddings)

	expectedResult := []float32{
		1, 2, 3,
		4, 5, 6,
		6, 7, 8,
	}

	cGradsValues := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}

	expectedEGrads := []float32{
		1, 2, 3,
		0, 0, 0,
		0, 0, 0,
		4, 5, 6,
		0, 0, 0,
		7, 8, 9,
	}

	copy(cData.Grad.GetFloats(), cGradsValues)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	cData.CalcData(b)
	cData.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedResult, cData.Data.GetFloats())
	require.Equal(t, expectedEGrads, tEmbeddings.Grad.GetFloats())
}

func TestDevice_Conv(t *testing.T) {
	device := NewWithSystemDefaultDevice()
	defer device.Release()

	input := device.NewDataWithValues(mtl.NewMTLSize(5, 5, 2), []float32{
		1, 2, 3, 4, 5,
		2, 3, 4, 5, 6,
		3, 4, 5, 6, 7,
		4, 5, 6, 7, 8,
		5, 6, 7, 8, 9,

		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		3, 3, 3, 3, 3,
		4, 4, 4, 4, 4,
		5, 5, 5, 5, 5,
	})

	weights := device.NewDataWithValues(mtl.NewMTLSize(3, 3, 2), []float32{
		1, 2, 3,
		2, 3, 4,
		4, 5, 6,

		1, 1, 1,
		2, 2, 2,
		3, 3, 3,
	})

	expectedOutputData := []float32{
		3 +
			1*1 + 2*2 + 3*3 +
			2*2 + 3*3 + 4*4 +
			3*4 + 4*5 + 5*6 +
			0 +
			1*1 + 1*1 + 1*1 +
			2*2 + 2*2 + 2*2 +
			3*3 + 3*3 + 3*3,

		3 +
			2*1 + 3*2 + 4*3 +
			3*2 + 4*3 + 5*4 +
			4*4 + 5*5 + 6*6 +
			0 +
			1*1 + 1*1 + 1*1 +
			2*2 + 2*2 + 2*2 +
			3*3 + 3*3 + 3*3,

		210, 198, 228, 258, 246, 276, 306,
	}

	biases := device.NewDataWithValues(mtl.NewMTLSize(1, 1, 1), []float32{
		3,
	})

	padding := 0
	stride := 1

	output := device.Conv(input, weights, biases, padding, stride)

	b := device.GetMTLDevice().NewCommandQueue().GetNewMTLCommandBuffer()
	defer b.Release()

	output.CalcData(b)
	output.CalcGrad(b)

	b.Commit()
	b.WaitUntilCompleted()

	require.Equal(t, expectedOutputData, output.Data.GetFloats())
}
