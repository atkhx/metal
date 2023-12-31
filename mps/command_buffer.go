package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void* mpsCommandBufferFromCommandQueue(void *commandQueueID) {
	return [MPSCommandBuffer commandBufferFromCommandQueue:(id<MTLCommandQueue>)commandQueueID];
}

void mpsCommandBufferRelease(void *mpsCommandBufferID) {
	[(__bridge MPSCommandBuffer*)mpsCommandBufferID release];
}

void* mpsCommandBufferGetMTLCommandBuffer(void *mpsCommandBufferID) {
	return [(__bridge MPSCommandBuffer*)mpsCommandBufferID commandBuffer];
}

void* mpsCommandBufferGetRootMTLCommandBuffer(void *mpsCommandBufferID) {
	return [(__bridge MPSCommandBuffer*)mpsCommandBufferID rootCommandBuffer];
}

void mpsCommandBufferCommitAndContinue(void *mpsCommandBufferID) {
	return [(__bridge MPSCommandBuffer*)mpsCommandBufferID commitAndContinue];
}

*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

type CommandBuffer struct {
	id unsafe.Pointer
}

func CommandBufferFromCommandQueue(commandQueue *mtl.CommandQueue) (*CommandBuffer, error) {
	id := unsafe.Pointer(C.mpsCommandBufferFromCommandQueue(commandQueue.GetID()))
	if id == nil {
		return nil, fmt.Errorf("mps command buffer id is nil")
	}
	return &CommandBuffer{id: id}, nil
}

func (b *CommandBuffer) Release() {
	C.mpsCommandBufferRelease(b.id)
}

func (b *CommandBuffer) GetID() unsafe.Pointer {
	return b.id
}

func (b *CommandBuffer) GetMTLCommandBufferID() unsafe.Pointer {
	return unsafe.Pointer(C.mpsCommandBufferGetMTLCommandBuffer(b.id))
}

// GetMTLCommandBuffer The Metal Command Buffer that was used to initialize this object.
func (b *CommandBuffer) GetMTLCommandBuffer() *mtl.CommandBuffer {
	return mtl.CreateCommandBuffer(b.GetMTLCommandBufferID())
}

func (b *CommandBuffer) GetRootMTLCommandBufferID() unsafe.Pointer {
	return unsafe.Pointer(C.mpsCommandBufferGetRootMTLCommandBuffer(b.id))
}

// GetRootMTLCommandBuffer The base MTLCommandBuffer underlying the CommandBuffer.
func (b *CommandBuffer) GetRootMTLCommandBuffer() *mtl.CommandBuffer {
	return mtl.CreateCommandBuffer(b.GetRootMTLCommandBufferID())
}

// CommitAndContinue Commit work encoded so far and continue with a new underlying command buffer.
func (b *CommandBuffer) CommitAndContinue() {
	C.mpsCommandBufferCommitAndContinue(b.id)
}
