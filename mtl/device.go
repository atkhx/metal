package mtl

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

void* mtlCreateSystemDefaultDevice() {
	return MTLCreateSystemDefaultDevice();
}

void mtlDeviceRelease(void *deviceID) {
    [(id<MTLDevice>)deviceID release];
}

const char* mtlDeviceGetName(void *deviceID) {
	return [[(id<MTLDevice>)deviceID name] UTF8String];
}

uint64_t mtlDeviceGetRegistryID(void *deviceID) {
	return [(id<MTLDevice>)deviceID registryID];
}

const char* mtlDeviceGetArchitecture(void *deviceID) {
	return [[[(id<MTLDevice>)deviceID architecture] name] UTF8String];
}

MTLSize mtlDeviceGetMaxThreadsPerThreadgroup(void *deviceID) {
   return [(id<MTLDevice>)deviceID maxThreadsPerThreadgroup];
}

bool mtlDeviceIsHeadless(void *deviceID) {
	return [(id<MTLDevice>)deviceID isHeadless];
}

bool mtlDeviceIsRemovable(void *deviceID) {
	return [(id<MTLDevice>)deviceID isRemovable];
}

bool mtlDeviceHasUnifiedMemory(void *deviceID) {
	return [(id<MTLDevice>)deviceID hasUnifiedMemory];
}

uint64_t mtlDeviceGetRecommendedMaxWorkingSetSize(void *deviceID) {
	return [(id<MTLDevice>)deviceID recommendedMaxWorkingSetSize];
}

MTLDeviceLocation mtlDeviceGetLocation(void *deviceID) {
	return [(id<MTLDevice>)deviceID location];
}

NSUInteger mtlDeviceGetLocationNumber(void *deviceID) {
	return [(id<MTLDevice>)deviceID locationNumber];
}

uint64_t mtlDeviceGetMaxTransferRate(void *deviceID) {
	return [(id<MTLDevice>)deviceID maxTransferRate];
}

bool mtlDeviceIsDepth24Stencil8PixelFormatSupported(void *deviceID) {
	return [(id<MTLDevice>)deviceID isDepth24Stencil8PixelFormatSupported];
}

MTLReadWriteTextureTier mtlDeviceGetReadWriteTextureSupport(void *deviceID) {
	return [(id<MTLDevice>)deviceID readWriteTextureSupport];
}

MTLArgumentBuffersTier mtlDeviceGetArgumentBuffersSupport(void *deviceID) {
	return [(id<MTLDevice>)deviceID argumentBuffersSupport];
}

bool mtlDeviceAreRasterOrderGroupsSupported(void *deviceID) {
	return [(id<MTLDevice>)deviceID areRasterOrderGroupsSupported];
}

bool mtlDeviceGetSupports32BitFloatFiltering(void *deviceID) {
	return [(id<MTLDevice>)deviceID supports32BitFloatFiltering];
}

bool mtlDeviceGetSupports32BitMSAA(void *deviceID) {
	return [(id<MTLDevice>)deviceID supports32BitMSAA];
}

bool mtlDeviceGetSupportsQueryTextureLOD(void *deviceID) {
	return [(id<MTLDevice>)deviceID supportsQueryTextureLOD];
}

bool mtlDeviceGetSupportsBCTextureCompression(void *deviceID) {
	return [(id<MTLDevice>)deviceID supportsBCTextureCompression];
}

bool mtlDeviceGetSupportsPullModelInterpolation(void *deviceID) {
	return [(id<MTLDevice>)deviceID supportsPullModelInterpolation];
}

bool mtlDeviceGetSupportsShaderBarycentricCoordinates(void *deviceID) {
	return [(id<MTLDevice>)deviceID supportsShaderBarycentricCoordinates];
}

NSUInteger mtlDeviceGetCurrentAllocatedSize(void *deviceID) {
	return [(id<MTLDevice>)deviceID currentAllocatedSize];
}

void* mtlDeviceNewCommandQueue(void *deviceID) {
    return [(id<MTLDevice>)deviceID newCommandQueue];
}

void* mtlDeviceNewBufferWithBytes(void *deviceID, float *bytes, NSUInteger length, MTLResourceOptions options) {
    return [(id<MTLDevice>)deviceID newBufferWithBytes:bytes length:length options:options];
}

void* mtlDeviceNewBufferWithLength(void *deviceID, size_t length, MTLResourceOptions options) {
    return [(id<MTLDevice>)deviceID
        newBufferWithLength:length
        options:options];
}

*/
import "C"
import (
	"fmt"
	"unsafe"
)

type Device struct {
	id unsafe.Pointer
}

func MustCreateSystemDefaultDevice() *Device {
	device, err := CreateSystemDefaultDevice()
	if err != nil {
		panic(err)
	}
	return device
}

func CreateSystemDefaultDevice() (*Device, error) {
	deviceID := unsafe.Pointer(C.mtlCreateSystemDefaultDevice())
	if deviceID == nil {
		return nil, fmt.Errorf("mtlCreateSystemDefaultDevice failed")
	}
	return &Device{id: deviceID}, nil
}

func (d *Device) Release() {
	C.mtlDeviceRelease(d.id)
}

func (d *Device) GetID() unsafe.Pointer {
	return d.id
}

func (d *Device) GetName() string {
	return C.GoString(C.mtlDeviceGetName(d.id))
}

func (d *Device) GetRegistryID() uint64 {
	return uint64(C.mtlDeviceGetRegistryID(d.id))
}

func (d *Device) GetArchitecture() string {
	return C.GoString(C.mtlDeviceGetArchitecture(d.id))
}

func (d *Device) GetMaxThreadsPerThreadgroup() MTLSize {
	return MTLSizeFromC(C.mtlDeviceGetMaxThreadsPerThreadgroup(d.id))
}

func (d *Device) IsHeadless() bool {
	return bool(C.mtlDeviceIsHeadless(d.id))
}

func (d *Device) IsRemovable() bool {
	return bool(C.mtlDeviceIsRemovable(d.id))
}

func (d *Device) HasUnifiedMemory() bool {
	return bool(C.mtlDeviceHasUnifiedMemory(d.id))
}

func (d *Device) GetRecommendedMaxWorkingSetSize() uint64 {
	return uint64(C.mtlDeviceGetRecommendedMaxWorkingSetSize(d.id))
}

func (d *Device) GetLocation() uint64 {
	return uint64(C.mtlDeviceGetLocation(d.id))
}

func (d *Device) GetLocationNumber() uint64 {
	return uint64(C.mtlDeviceGetLocationNumber(d.id))
}

func (d *Device) GetMaxTransferRate() uint64 {
	return uint64(C.mtlDeviceGetMaxTransferRate(d.id))
}

func (d *Device) IsDepth24Stencil8PixelFormatSupported() bool {
	return bool(C.mtlDeviceIsDepth24Stencil8PixelFormatSupported(d.id))
}

func (d *Device) GetReadWriteTextureSupport() uint64 {
	return uint64(C.mtlDeviceGetReadWriteTextureSupport(d.id))
}

func (d *Device) GetArgumentBuffersSupport() uint64 {
	return uint64(C.mtlDeviceGetArgumentBuffersSupport(d.id))
}

func (d *Device) AreRasterOrderGroupsSupported() bool {
	return bool(C.mtlDeviceAreRasterOrderGroupsSupported(d.id))
}

func (d *Device) GetSupports32BitFloatFiltering() bool {
	return bool(C.mtlDeviceGetSupports32BitFloatFiltering(d.id))
}

func (d *Device) GetSupports32BitMSAA() bool {
	return bool(C.mtlDeviceGetSupports32BitMSAA(d.id))
}

func (d *Device) GetSupportsQueryTextureLOD() bool {
	return bool(C.mtlDeviceGetSupportsQueryTextureLOD(d.id))
}

func (d *Device) GetSupportsBCTextureCompression() bool {
	return bool(C.mtlDeviceGetSupportsBCTextureCompression(d.id))
}

func (d *Device) GetSupportsPullModelInterpolation() bool {
	return bool(C.mtlDeviceGetSupportsPullModelInterpolation(d.id))
}

func (d *Device) GetSupportsShaderBarycentricCoordinates() bool {
	return bool(C.mtlDeviceGetSupportsShaderBarycentricCoordinates(d.id))
}

func (d *Device) GetCurrentAllocatedSize() uint64 {
	return uint64(C.mtlDeviceGetCurrentAllocatedSize(d.id))
}

func (d *Device) NewCommandQueue() *CommandQueue {
	return CreateCommandQueue(unsafe.Pointer(C.mtlDeviceNewCommandQueue(d.id)))
}

func (d *Device) NewBufferWithBytes(data []byte, options resourceOptions) *Buffer {
	return CreateBuffer(C.mtlDeviceNewBufferWithBytes(
		d.id,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.ulong(len(data)),
		C.MTLResourceOptions(options),
	))
}

func (d *Device) NewBufferWithFloats(data []float32, options resourceOptions) *Buffer {
	return CreateBuffer(C.mtlDeviceNewBufferWithBytes(
		d.id,
		(*C.float)(unsafe.Pointer(&data[0])),
		C.ulong(len(data)*int(unsafe.Sizeof(float32(0)))),
		C.MTLResourceOptions(options),
	))
}

func (d *Device) NewBufferEmptyFloatsBuffer(length int, options resourceOptions) *Buffer {
	return CreateBuffer(C.mtlDeviceNewBufferWithLength(
		d.id,
		C.ulong(length*int(unsafe.Sizeof(float32(0)))),
		C.MTLResourceOptions(options),
	))
}
