package mps

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void* mpsImageDescriptorCreateFloat32(int width, int height, int channels, int batchSize) {
    return [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                         width:(NSUInteger)width
                                                        height:(NSUInteger)height
                                               featureChannels:(NSUInteger)channels
                                                numberOfImages:(NSUInteger)batchSize
                                                         usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
}

void* mpsImageCreate(void *deviceID, void *descriptorID) {
    return [[MPSImage alloc] initWithDevice:(id<MTLDevice>)deviceID
                            imageDescriptor:(MPSImageDescriptor*)descriptorID];
}

void* mpsImageGetTexture(void *imageID) {
    return [(__bridge MPSImage*)imageID texture];
}

NSUInteger mpsImageGetWidth(void *imageID) {
    return [(__bridge MPSImage*)imageID width];
}

NSUInteger mpsImageGetHeight(void *imageID) {
    return [(__bridge MPSImage*)imageID height];
}

NSUInteger mpsImageGetFeatureChannels(void *imageID) {
    return [(__bridge MPSImage*)imageID featureChannels];
}

NSUInteger mpsImageGetNumberOfImages(void *imageID) {
    return [(__bridge MPSImage*)imageID numberOfImages];
}

void mpsImageRelease(void *imageID) {
    [(__bridge MPSImage*)imageID release];
}

void mpsImageDescriptorRelease(void *descriptorID) {
    [(__bridge MPSImageDescriptor*)descriptorID release];
}

*/
import "C"
import (
	"unsafe"

	"github.com/atkhx/metal/mtl"
)

type Image struct {
	id unsafe.Pointer
}

func NewImageFloat32(device *mtl.Device, width, height, channels, batch int) *Image {
	desc := unsafe.Pointer(C.mpsImageDescriptorCreateFloat32(
		C.int(width),
		C.int(height),
		C.int(channels),
		C.int(batch),
	))
	defer C.mpsImageDescriptorRelease(desc)

	id := unsafe.Pointer(C.mpsImageCreate(device.GetID(), desc))
	if id == nil {
		panic("MPSImage: id is empty")
	}
	return &Image{id: id}
}

func (i *Image) Release() {
	C.mpsImageRelease(i.id)
}

func (i *Image) GetID() unsafe.Pointer {
	return i.id
}

func (i *Image) GetTextureID() unsafe.Pointer {
	return unsafe.Pointer(C.mpsImageGetTexture(i.id))
}

func (i *Image) GetWidth() int {
	return int(C.mpsImageGetWidth(i.id))
}

func (i *Image) GetHeight() int {
	return int(C.mpsImageGetHeight(i.id))
}

func (i *Image) GetFeatureChannels() int {
	return int(C.mpsImageGetFeatureChannels(i.id))
}

func (i *Image) GetNumberOfImages() int {
	return int(C.mpsImageGetNumberOfImages(i.id))
}
