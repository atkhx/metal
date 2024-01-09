package model

import (
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/layer"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/proc"

	jsoniter "github.com/json-iterator/go"
)

var json = jsoniter.ConfigCompatibleWithStandardLibrary

func New(
	inDims mtl.MTLSize,
	layers layer.Layers,
	device *proc.Device,
	optimizer proc.Optimizer,
) *Model {
	return &Model{
		inDims:    inDims,
		Layers:    layers,
		device:    device,
		optimizer: optimizer,
	}
}

type Model struct {
	inDims mtl.MTLSize
	inputs *num.Data
	output *num.Data
	Layers layer.Layers
	update []*num.Data
	device *proc.Device

	optimizer  proc.Optimizer
	updateFunc func(b *mtl.CommandBuffer, iteration int)
}

func (s *Model) Compile() *num.Data {
	s.inputs = s.device.NewData(s.inDims)
	s.output = s.Layers.Compile(s.device, s.inputs)

	s.update = append(s.update, s.Layers.ForUpdate()...)
	if s.optimizer != nil {
		s.updateFunc = s.optimizer(s.update)
	}

	return s.output
}

func (s *Model) GetInput() *num.Data {
	return s.inputs
}

func (s *Model) GetOutput() *num.Data {
	return s.output
}

func (s *Model) GetTrainableParamsCount() (result int) {
	for _, node := range s.update {
		result += node.Dims.Length()
	}
	return result
}

func (s *Model) Update(b *mtl.CommandBuffer, iteration int) {
	s.updateFunc(b, iteration)
}

func (s *Model) LoadFromFile(filename string) error {
	t := time.Now()
	config, err := os.ReadFile(filename)
	if err != nil && errors.Is(err, os.ErrNotExist) {
		log.Println("trained config not found (skip)")
		return nil
	}
	if err != nil {
		return fmt.Errorf("read file failed: %w", err)
	}
	fmt.Println("read config success:", time.Since(t))

	t = time.Now()
	if err = json.Unmarshal(config, s); err != nil {
		return fmt.Errorf("unmarshal config failed: %w", err)
	}
	fmt.Println("unmarshal success:", time.Since(t))
	return nil
}

func (s *Model) LoadFromProvider() {
	for _, ll := range s.Layers {
		if l, ok := ll.(layer.WithWeightsProvider); ok {
			l.LoadFromProvider()
		}
	}
}

func (s *Model) SaveToFile(filename string) error {
	t := time.Now()
	nnBytes, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("marshal model config failed: %w", err)
	}

	fmt.Println("marshal success:", time.Since(t))

	t = time.Now()
	if err := os.WriteFile(filename, nnBytes, os.ModePerm); err != nil {
		return fmt.Errorf("write model config failed: %w", err)
	}
	fmt.Println("save success:", time.Since(t))
	return nil
}
