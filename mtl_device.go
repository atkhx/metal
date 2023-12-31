package metal

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework CoreGraphics -framework Foundation

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

typedef MTLSize CMTLSize;

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

CMTLSize mtlDeviceGetMaxThreadsPerThreadgroup(void *deviceID) {
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

// Methods

void* mtlDeviceNewCommandQueue(void *deviceID) {
    return [(id<MTLDevice>)deviceID newCommandQueue];
}

*/
import "C"
import (
	"fmt"
	"unsafe"
)

type MTLSize struct {
	W, H, D uint64
}

func ToMTLSize(r C.CMTLSize) MTLSize {
	return MTLSize{
		W: uint64(r.width),
		H: uint64(r.height),
		D: uint64(r.depth),
	}
}

type MTLDevice struct {
	deviceID unsafe.Pointer
}

func MustMTLCreateSystemDefaultDevice() *MTLDevice {
	device, err := MTLCreateSystemDefaultDevice()
	if err != nil {
		panic(err)
	}
	return device
}

func MTLCreateSystemDefaultDevice() (*MTLDevice, error) {
	deviceID := unsafe.Pointer(C.mtlCreateSystemDefaultDevice())
	if deviceID == nil {
		return nil, fmt.Errorf("mtlCreateSystemDefaultDevice failed")
	}
	return &MTLDevice{deviceID: deviceID}, nil
}

func (d *MTLDevice) Release() {
	C.mtlDeviceRelease(d.deviceID)
}

func (d *MTLDevice) GetDeviceID() unsafe.Pointer {
	return d.deviceID
}

func (d *MTLDevice) GetName() string {
	return C.GoString(C.mtlDeviceGetName(d.deviceID))
}

func (d *MTLDevice) GetRegistryID() uint64 {
	return uint64(C.mtlDeviceGetRegistryID(d.deviceID))
}

func (d *MTLDevice) GetArchitecture() string {
	return C.GoString(C.mtlDeviceGetArchitecture(d.deviceID))
}

func (d *MTLDevice) GetMaxThreadsPerThreadgroup() MTLSize {
	return ToMTLSize(C.mtlDeviceGetMaxThreadsPerThreadgroup(d.deviceID))
}

func (d *MTLDevice) IsHeadless() bool {
	return bool(C.mtlDeviceIsHeadless(d.deviceID))
}

func (d *MTLDevice) IsRemovable() bool {
	return bool(C.mtlDeviceIsRemovable(d.deviceID))
}

func (d *MTLDevice) HasUnifiedMemory() bool {
	return bool(C.mtlDeviceHasUnifiedMemory(d.deviceID))
}

func (d *MTLDevice) GetRecommendedMaxWorkingSetSize() uint64 {
	return uint64(C.mtlDeviceGetRecommendedMaxWorkingSetSize(d.deviceID))
}

func (d *MTLDevice) GetLocation() uint64 {
	return uint64(C.mtlDeviceGetLocation(d.deviceID))
}

func (d *MTLDevice) GetLocationNumber() uint64 {
	return uint64(C.mtlDeviceGetLocationNumber(d.deviceID))
}

func (d *MTLDevice) GetMaxTransferRate() uint64 {
	return uint64(C.mtlDeviceGetMaxTransferRate(d.deviceID))
}

func (d *MTLDevice) IsDepth24Stencil8PixelFormatSupported() bool {
	return bool(C.mtlDeviceIsDepth24Stencil8PixelFormatSupported(d.deviceID))
}

func (d *MTLDevice) GetReadWriteTextureSupport() uint64 {
	return uint64(C.mtlDeviceGetReadWriteTextureSupport(d.deviceID))
}

func (d *MTLDevice) GetArgumentBuffersSupport() uint64 {
	return uint64(C.mtlDeviceGetArgumentBuffersSupport(d.deviceID))
}

func (d *MTLDevice) AreRasterOrderGroupsSupported() bool {
	return bool(C.mtlDeviceAreRasterOrderGroupsSupported(d.deviceID))
}

func (d *MTLDevice) GetSupports32BitFloatFiltering() bool {
	return bool(C.mtlDeviceGetSupports32BitFloatFiltering(d.deviceID))
}

func (d *MTLDevice) GetSupports32BitMSAA() bool {
	return bool(C.mtlDeviceGetSupports32BitMSAA(d.deviceID))
}

func (d *MTLDevice) GetSupportsQueryTextureLOD() bool {
	return bool(C.mtlDeviceGetSupportsQueryTextureLOD(d.deviceID))
}

func (d *MTLDevice) GetSupportsBCTextureCompression() bool {
	return bool(C.mtlDeviceGetSupportsBCTextureCompression(d.deviceID))
}

func (d *MTLDevice) GetSupportsPullModelInterpolation() bool {
	return bool(C.mtlDeviceGetSupportsPullModelInterpolation(d.deviceID))
}

func (d *MTLDevice) GetSupportsShaderBarycentricCoordinates() bool {
	return bool(C.mtlDeviceGetSupportsShaderBarycentricCoordinates(d.deviceID))
}

func (d *MTLDevice) GetCurrentAllocatedSize() uint64 {
	return uint64(C.mtlDeviceGetCurrentAllocatedSize(d.deviceID))
}

// Methods

func (d *MTLDevice) NewCommandQueue() (*MTLCommandQueue, error) {
	return MTLCommandQueueCreate(unsafe.Pointer(C.mtlDeviceNewCommandQueue(d.deviceID)))
}
