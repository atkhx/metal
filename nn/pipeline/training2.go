package pipeline

import (
	"github.com/atkhx/metal/mps"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/num"
	"github.com/atkhx/metal/nn/ops/fill"
)

func NewTrainingPipeline2(device *mtl.Device, lastNode *num.Data) *TrainingPipeline2 {
	return &TrainingPipeline2{
		device:          device,
		forwardLayers:   getForwardNodeLayers(lastNode),
		backwardLayers:  getBackwardNodeLayers(lastNode),
		resetGradsNodes: getResetGradsNodes(lastNode),
		commandQueue:    device.NewCommandQueue(),
	}
}

type TrainingPipeline2 struct {
	device *mtl.Device

	forwardLayers   NodeLayers
	backwardLayers  NodeLayers
	resetGradsNodes Nodes
	commandQueue    *mtl.CommandQueue
}

func (p *TrainingPipeline2) TrainIteration(update func(b *mtl.CommandBuffer)) {
	mpsCommandBuffer, err := mps.CommandBufferFromCommandQueue(p.commandQueue)
	if err != nil {
		panic(err)
	}

	for _, nodes := range p.forwardLayers {
		for _, node := range nodes {
			node.CalcData(mpsCommandBuffer.GetMTLCommandBuffer())
			mpsCommandBuffer.CommitAndContinue()
		}
	}

	fillKernel := fill.New(p.device)
	for i, node := range p.resetGradsNodes {
		b := mpsCommandBuffer.GetMTLCommandBuffer()
		if i == 0 {
			fillKernel.Fill(b, node.Grad, 1.0, 0, node.Dims.Length())
		} else {
			fillKernel.Fill(b, node.Grad, 0.0, 0, node.Dims.Length())
		}
		mpsCommandBuffer.CommitAndContinue()
	}

	for _, nodes := range p.backwardLayers {
		for _, node := range nodes {
			node.CalcGrad(mpsCommandBuffer.GetMTLCommandBuffer())
			mpsCommandBuffer.CommitAndContinue()
		}
	}

	b := mpsCommandBuffer.GetMTLCommandBuffer()
	update(b)
	b.Commit()

	rootCommandBuffer := mpsCommandBuffer.GetRootMTLCommandBuffer()
	rootCommandBuffer.WaitUntilCompleted()
}
