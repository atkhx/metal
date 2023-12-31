package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>

void mtlCommandQueueRelease(void *commandQueueID) {
    [(id<MTLCommandQueue>)commandQueueID release];
}

*/
import "C"
import (
	"fmt"
	"unsafe"
)

type MTLCommandQueue struct {
	id unsafe.Pointer
}

func MustCreateMTLCommandQueue(id unsafe.Pointer) *MTLCommandQueue {
	q, err := MTLCommandQueueCreate(id)
	if err != nil {
		panic(err)
	}
	return q
}

func MTLCommandQueueCreate(id unsafe.Pointer) (*MTLCommandQueue, error) {
	if id == nil {
		return nil, fmt.Errorf("command queue id is null")
	}
	return &MTLCommandQueue{id: id}, nil
}

func (q *MTLCommandQueue) Release() {
	C.mtlCommandQueueRelease(q.id)
}
