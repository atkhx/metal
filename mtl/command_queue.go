package mtl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>

void mtlCommandQueueRelease(void *commandQueueID) {
    [(id<MTLCommandQueue>)commandQueueID release];
}

void* mtlCommandQueueGetNewMTLCommandBuffer(void *commandQueueID) {
    return [(id<MTLCommandQueue>)commandQueueID commandBuffer];
}

*/
import "C"
import (
	"unsafe"
)

type CommandQueue struct {
	id unsafe.Pointer
}

func CreateCommandQueue(id unsafe.Pointer) *CommandQueue {
	if id == nil {
		panic("MTLCommandQueue: empty id")
	}
	return &CommandQueue{id: id}
}

func (q *CommandQueue) Release() {
	C.mtlCommandQueueRelease(q.id)
}

func (q *CommandQueue) GetID() unsafe.Pointer {
	return q.id
}

func (q *CommandQueue) GetNewMTLCommandBuffer() *CommandBuffer {
	return CreateCommandBuffer(C.mtlCommandQueueGetNewMTLCommandBuffer(q.id))
}
