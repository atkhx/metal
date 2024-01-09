package pipeline

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/ops/fill"
)

func NewTrainingPipeline(device *mtl.Device, lastNode *num.Data) *TrainingPipeline {
	return &TrainingPipeline{
		device:          device,
		forwardLayers:   getForwardNodeLayers(lastNode),
		backwardLayers:  getBackwardNodeLayers(lastNode),
		resetGradsNodes: getResetGradsNodes(lastNode),
		commandQueue:    device.NewCommandQueue(),
	}
}

type TrainingPipeline struct {
	device *mtl.Device

	forwardLayers   NodeLayers
	backwardLayers  NodeLayers
	resetGradsNodes Nodes
	commandQueue    *mtl.CommandQueue
}

func (p *TrainingPipeline) withCommandBuffer(callback func(b *mtl.CommandBuffer)) {
	commandBuffer := p.commandQueue.GetNewMTLCommandBuffer()
	defer commandBuffer.Release()
	callback(commandBuffer)
	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()
}

func (p *TrainingPipeline) forward(b *mtl.CommandBuffer) {
	for _, nodes := range p.forwardLayers {
		for _, node := range nodes {
			node.CalcData(b)
		}
	}
}

func (p *TrainingPipeline) reset(b *mtl.CommandBuffer) {
	fillKernel := fill.New(p.device)
	for i, node := range p.resetGradsNodes {
		if i == 0 {
			fillKernel.Fill(b, node.Grad, 1.0, 0, node.Dims.Length())
		} else {
			fillKernel.Fill(b, node.Grad, 0.0, 0, node.Dims.Length())
		}
	}
}

func (p *TrainingPipeline) backward(b *mtl.CommandBuffer) {
	for _, nodes := range p.backwardLayers {
		for _, node := range nodes {
			node.CalcGrad(b)
		}
	}
}

func (p *TrainingPipeline) TrainIteration(update func(b *mtl.CommandBuffer)) {
	p.withCommandBuffer(func(b *mtl.CommandBuffer) {
		p.forward(b)
		p.reset(b)
		p.backward(b)
		update(b)
	})
}
