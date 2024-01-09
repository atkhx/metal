package layer

import (
	"encoding/json"

	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"
)

func NewEmbeddings(
	embeddings *num.Data,
	provideWeights func(embeddings *num.Data),
) *Embeddings {
	return &Embeddings{
		embeddings:     embeddings,
		provideWeights: provideWeights,
		forUpdate:      []*num.Data{embeddings},
	}
}

type Embeddings struct {
	embeddings     *num.Data
	forUpdate      []*num.Data
	provideWeights func(embeddings *num.Data)
}

func (l *Embeddings) Compile(device *proc.Device, inputs *num.Data) *num.Data {
	return device.Embeddings(inputs, l.embeddings)
}

func (l *Embeddings) ForUpdate() []*num.Data {
	return l.forUpdate
}

func (l *Embeddings) MarshalJSON() ([]byte, error) {
	return json.Marshal(l.embeddings.Data.GetFloats())
}

func (l *Embeddings) UnmarshalJSON(bytes []byte) error {
	weights := l.embeddings.Data.GetFloats()
	return json.Unmarshal(bytes, &weights)
}

func (l *Embeddings) LoadFromProvider() {
	l.provideWeights(l.embeddings)
}
