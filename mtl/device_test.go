package mtl

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMTLDeviceProperties(t *testing.T) {
	device := MustCreateSystemDefaultDevice()
	defer device.Release()

	require.NotNil(t, device.GetID())

	deviceName := device.GetName()
	t.Log("deviceName", deviceName)
	require.NotEmpty(t, deviceName)

	registryID := device.GetRegistryID()
	t.Log("registryID", registryID)
	require.NotEmpty(t, registryID)

	architecture := device.GetArchitecture()
	t.Log("architecture", architecture)
	require.NotEmpty(t, architecture)

	maxThreadsPerThreadgroup := device.GetMaxThreadsPerThreadgroup()
	t.Log("maxThreadsPerThreadgroup", maxThreadsPerThreadgroup)

	require.Greater(t, maxThreadsPerThreadgroup.W, 0)
	require.Greater(t, maxThreadsPerThreadgroup.H, 0)
	require.Greater(t, maxThreadsPerThreadgroup.D, 0)

	t.Log("isHeadless", device.IsHeadless())
	t.Log("isRemovable", device.IsRemovable())
	t.Log("hasUnifiedMemory", device.HasUnifiedMemory())
	t.Log("recommendedMaxWorkingSetSize", device.GetRecommendedMaxWorkingSetSize())
	t.Log("recommendedMaxWorkingSetSize kB", device.GetRecommendedMaxWorkingSetSize()/1024)
	t.Log("recommendedMaxWorkingSetSize mB", device.GetRecommendedMaxWorkingSetSize()/1024/1024)
	t.Log("recommendedMaxWorkingSetSize GB", device.GetRecommendedMaxWorkingSetSize()/1024/1024/1024)
	t.Log("location code", device.GetLocation())
	t.Log("location number", device.GetLocationNumber())
	t.Log("maxTransferRate", device.GetMaxTransferRate())
	t.Log("isDepth24Stencil8PixelFormatSupported", device.IsDepth24Stencil8PixelFormatSupported())
	t.Log("readWriteTextureSupport", device.GetReadWriteTextureSupport())
	t.Log("argumentBuffersSupport", device.GetArgumentBuffersSupport())
	t.Log("areRasterOrderGroupsSupported", device.AreRasterOrderGroupsSupported())
	t.Log("supports32BitFloatFiltering", device.GetSupports32BitFloatFiltering())
	t.Log("supports32BitMSAA", device.GetSupports32BitMSAA())
	t.Log("supportsQueryTextureLOD", device.GetSupportsQueryTextureLOD())
	t.Log("supportsBCTextureCompression", device.GetSupportsBCTextureCompression())
	t.Log("supportsPullModelInterpolation", device.GetSupportsPullModelInterpolation())
	t.Log("supportsShaderBarycentricCoordinates", device.GetSupportsShaderBarycentricCoordinates())
	t.Log("currentAllocatedSize", device.GetCurrentAllocatedSize())
}

func TestMTLDevice_NewCommandQueue(t *testing.T) {
	device := MustCreateSystemDefaultDevice()
	require.NotNil(t, device.GetID())
	defer device.Release()

	commandQueue := device.NewCommandQueue()
	defer commandQueue.Release()
}
