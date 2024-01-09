package pipeline

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/ops/fill"
)

func NewTestingPipeline(device *mtl.Device, lastNode *num.Data) *TestingPipeline {
	return &TestingPipeline{
		device:          device,
		forwardLayers:   getForwardNodeLayers(lastNode),
		backwardLayers:  getBackwardNodeLayers(lastNode),
		resetGradsNodes: getResetGradsNodes(lastNode),
		commandQueue:    device.NewCommandQueue(),
	}
}

type TestingPipeline struct {
	device *mtl.Device

	forwardLayers   NodeLayers
	backwardLayers  NodeLayers
	resetGradsNodes Nodes
	commandQueue    *mtl.CommandQueue
}

func (p *TestingPipeline) withCommandBuffer(callback func(b *mtl.CommandBuffer)) {
	commandBuffer := p.commandQueue.GetNewMTLCommandBuffer()
	defer commandBuffer.Release()
	callback(commandBuffer)
	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()
}

func (p *TestingPipeline) Forward() {
	p.withCommandBuffer(p.forward)
}

func (p *TestingPipeline) Reset() {
	p.withCommandBuffer(p.reset)
}

func (p *TestingPipeline) Backward() {
	p.withCommandBuffer(p.backward)
}

func (p *TestingPipeline) forward(b *mtl.CommandBuffer) {
	for _, nodes := range p.forwardLayers {
		for _, node := range nodes {
			node.CalcData(b)
		}
	}
}

func (p *TestingPipeline) reset(b *mtl.CommandBuffer) {
	fillKernel := fill.New(p.device)
	for i, node := range p.resetGradsNodes {
		if i == 0 {
			fillKernel.Fill(b, node.Grad, 1.0, 0, node.Dims.Length())
		} else {
			fillKernel.Fill(b, node.Grad, 0.0, 0, node.Dims.Length())
		}
	}
}

func (p *TestingPipeline) backward(b *mtl.CommandBuffer) {
	for _, nodes := range p.backwardLayers {
		for _, node := range nodes {
			node.CalcGrad(b)
		}
	}
}
