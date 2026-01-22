package initializer

type InitWeightFixed struct {
	NormK float32
}

func (wi InitWeightFixed) GetNormK(_, _ int) float32 {
	return wi.NormK
}
