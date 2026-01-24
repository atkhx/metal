package layer

import (
	"math/rand"
	"testing"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/initializer"
	"github.com/atkhx/metal/nn/proc"
	"github.com/stretchr/testify/require"
)

type testInit struct {
	k    float32
	dist initializer.Distribution
}

func (t testInit) GetNormK(_, _ int) float32 {
	return t.k
}

func (t testInit) Distribution() initializer.Distribution {
	return t.dist
}

func TestInitWeights_Distribution(t *testing.T) {
	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	rand.Seed(1)
	k := float32(0.1)

	normal := initWeights(device, testInit{k: k, dist: initializer.DistributionNormal}, mtl.NewMTLSize(1024), 1, 1)
	uniform := initWeights(device, testInit{k: k, dist: initializer.DistributionUniform}, mtl.NewMTLSize(1024), 1, 1)

	maxAbsNormal := float32(0)
	maxAbsUniform := float32(0)

	for _, v := range normal.Data.GetFloats() {
		av := v
		if av < 0 {
			av = -av
		}
		if av > maxAbsNormal {
			maxAbsNormal = av
		}
	}

	for _, v := range uniform.Data.GetFloats() {
		av := v
		if av < 0 {
			av = -av
		}
		if av > maxAbsUniform {
			maxAbsUniform = av
		}
	}

	require.Greater(t, maxAbsNormal, k)
	require.LessOrEqual(t, maxAbsUniform, k)
}
