package initializer

type InitWeightFixed struct {
	NormK float32
}

func (wi InitWeightFixed) GetNormK(_, _ int) float32 {
	return wi.NormK
}

func (wi InitWeightFixed) Distribution() Distribution {
	return DistributionUniform
}

// InitWeightFixedNormal uses N(0, NormK^2).
type InitWeightFixedNormal struct {
	NormK float32
}

func (wi InitWeightFixedNormal) GetNormK(_, _ int) float32 {
	return wi.NormK
}

func (wi InitWeightFixedNormal) Distribution() Distribution {
	return DistributionNormal
}
