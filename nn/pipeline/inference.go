package pipeline

import (
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
)

func NewInferencePipeline(device *mtl.Device, lastNode *num.Data) *InferencePipeline {
	return &InferencePipeline{
		device:        device,
		forwardLayers: getForwardNodeLayers(lastNode),
		commandQueue:  device.NewCommandQueue(),
	}
}

type InferencePipeline struct {
	device *mtl.Device

	forwardLayers NodeLayers
	commandQueue  *mtl.CommandQueue
}

func (p *InferencePipeline) withCommandBuffer(callback func(b *mtl.CommandBuffer)) {
	commandBuffer := p.commandQueue.GetNewMTLCommandBuffer()
	defer commandBuffer.Release()
	callback(commandBuffer)
	commandBuffer.Commit()
	commandBuffer.WaitUntilCompleted()
}

func (p *InferencePipeline) Forward() {
	p.withCommandBuffer(func(b *mtl.CommandBuffer) {
		for _, nodes := range p.forwardLayers {
			for _, node := range nodes {
				node.CalcData(b)
			}
		}
	})
}
