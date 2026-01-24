package adamw

import (
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/stretchr/testify/require"
)

func TestAdamW_UsesEps(t *testing.T) {
	device := mtl.MustCreateSystemDefaultDevice()
	defer device.Release()

	kernel := New(device)

	run := func(eps float32) float32 {
		data := device.NewBufferWithFloats([]float32{1}, mtl.ResourceStorageModeShared)
		grad := device.NewBufferWithFloats([]float32{1}, mtl.ResourceStorageModeShared)
		m := device.NewBufferWithFloats([]float32{0}, mtl.ResourceStorageModeShared)
		v := device.NewBufferWithFloats([]float32{0}, mtl.ResourceStorageModeShared)

		cmd := device.NewCommandQueue().GetNewMTLCommandBuffer()
		defer cmd.Release()

		kernel.UpdateWithAdam(cmd, data, grad, m, v, 0, 0, 1, 1, eps)
		cmd.Commit()
		cmd.WaitUntilCompleted()

		return data.GetFloats()[0]
	}

	outSmall := run(1e-2)
	outLarge := run(1e-1)

	expectedSmall := float32(1 - 1/(1+1e-2))
	expectedLarge := float32(1 - 1/(1+1e-1))

	require.InDelta(t, float64(expectedSmall), float64(outSmall), 1e-5)
	require.InDelta(t, float64(expectedLarge), float64(outLarge), 1e-5)
	require.NotEqual(t, outSmall, outLarge)
}
