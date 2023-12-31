package mtl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void* mtlCommandBufferGetCommandQueue(void *commandBufferID) {
	return [(id<MTLCommandBuffer>)commandBufferID commandQueue];
}

void mtlCommandBufferRelease(void *commandBufferID) {
	[(id<MTLCommandBuffer>)commandBufferID release];
}

bool mtlCommandBufferIsRetainedReferences(void *commandBufferID) {
	return [(id<MTLCommandBuffer>)commandBufferID retainedReferences];
}

void mtlCommandBufferEnqueue(void *commandBufferID) {
	[(id<MTLCommandBuffer>)commandBufferID enqueue];
}

void mtlCommandBufferCommit(void *commandBufferID) {
	[(id<MTLCommandBuffer>)commandBufferID commit];
}

void mtlCommandBufferWaitUntilScheduled(void *commandBufferID) {
	[(id<MTLCommandBuffer>)commandBufferID waitUntilScheduled];
}

void mtlCommandBufferWaitUntilCompleted(void *commandBufferID) {
	[(id<MTLCommandBuffer>)commandBufferID waitUntilCompleted];
}

MTLCommandBufferStatus mtlCommandBufferGetStatus(void *commandBufferID) {
	return [(id<MTLCommandBuffer>)commandBufferID status];
}

void* mtlCommandBufferGetMTLBlitCommandEncoder(void *commandBufferID) {
	return [(id<MTLCommandBuffer>)commandBufferID blitCommandEncoder];
}

*/
import "C"
import (
	"unsafe"
)

type CommandBuffer struct {
	id unsafe.Pointer
}

func CreateCommandBuffer(id unsafe.Pointer) *CommandBuffer {
	if id == nil {
		panic("MTLCommandBuffer: id is empty")
	}
	return &CommandBuffer{id: id}
}

func (b *CommandBuffer) Release() {
	C.mtlCommandBufferRelease(b.id)
}

func (b *CommandBuffer) GetID() unsafe.Pointer {
	return b.id
}

func (b *CommandBuffer) GetCommandQueue() *CommandQueue {
	return CreateCommandQueue(C.mtlCommandBufferGetCommandQueue(b.id))
}

func (b *CommandBuffer) IsRetainedReferences() bool {
	return bool(C.mtlCommandBufferIsRetainedReferences(b.id))
}

// Enqueue Append this command buffer to the end of its MTLCommandQueue
func (b *CommandBuffer) Enqueue() {
	C.mtlCommandBufferEnqueue(b.id)
}

// Commit Commits a command buffer, so it can be executed as soon as possible.
func (b *CommandBuffer) Commit() {
	C.mtlCommandBufferCommit(b.id)
}

// WaitUntilScheduled Synchronously wait for this command buffer to be scheduled.
func (b *CommandBuffer) WaitUntilScheduled() {
	C.mtlCommandBufferWaitUntilScheduled(b.id)
}

// WaitUntilCompleted Synchronously wait for this command buffer to complete.
func (b *CommandBuffer) WaitUntilCompleted() {
	C.mtlCommandBufferWaitUntilCompleted(b.id)
}

// GetStatus Status reports the current stage in the lifetime of MTLCommandBuffer,
// as it proceeds to enqueued, committed, scheduled, and completed.
func (b *CommandBuffer) GetStatus() uint64 {
	return uint64(C.mtlCommandBufferGetStatus(b.id))
}

func (b *CommandBuffer) GetMTLBlitCommandEncoderID() unsafe.Pointer {
	return unsafe.Pointer(C.mtlCommandBufferGetMTLBlitCommandEncoder(b.id))
}

// GetMTLBlitCommandEncoder returns a blit command encoder to encode into this command buffer.
func (b *CommandBuffer) GetMTLBlitCommandEncoder() *BlitCommandEncoder {
	return CreateBlitCommandEncoder(b.GetMTLBlitCommandEncoderID())
}
