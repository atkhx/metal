package dataset

type Sample struct {
	Input, Target []float32
}

type Dataset interface {
	GetSamplesCount() int
	ReadSample(index int) (sample Sample, err error)
	ReadRandomSampleBatch(batchSize int) (sample Sample, err error)
}

type ClassifierDataset interface {
	Dataset

	GetClasses() []string
}
