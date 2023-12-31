package mtl

// purgeableState Options for setPurgeable call
type purgeableState uint64

const (
	PurgeableStateKeepCurrent purgeableState = 1
	PurgeableStateNonVolatile purgeableState = 2
	PurgeableStateVolatile    purgeableState = 3
	PurgeableStateEmpty       purgeableState = 4
)

// cpuCacheMode Describes what CPU cache mode is used for the CPU's mapping of a texture resource.
type cpuCacheMode uint64

const (
	cpuCacheModeDefaultCache  cpuCacheMode = 0
	cpuCacheModeWriteCombined cpuCacheMode = 1
)

// storageMode Describes location and CPU mapping of MTLTexture.
type storageMode uint64

const (
	// storageModeShared In this mode, CPU and device will nominally both use the same underlying memory when accessing the contents of the texture resource.
	// However, coherency is only guaranteed at command buffer boundaries to minimize the required flushing of CPU and GPU caches.
	// This is the default storage mode for iOS Textures.
	storageModeShared storageMode = 0

	// storageModeManaged This mode relaxes the coherency requirements and requires that the developer make explicit requests to maintain
	// coherency between a CPU and GPU version of the texture resource.  In order for CPU to access up-to-date GPU results,
	// first, a blit synchronizations must be completed (see synchronize methods of MTLBlitCommandEncoder).
	// Blit overhead is only incurred if GPU has modified the resource.
	// This is the default storage mode for OS X Textures.
	storageModeManaged storageMode = 1 // API_UNAVAILABLE(ios)

	// storageModePrivate This mode allows the texture resource data to be kept entirely to GPU (or driver) private memory that will never be accessed by the CPU directly, so no
	// coherency of any kind must be maintained.
	storageModePrivate storageMode = 2

	// storageModeMemoryless This mode allows creation of resources that do not have a GPU or CPU memory backing, but do have on-chip storage for TBDR
	// devices. The contents of the on-chip storage is undefined and does not persist, but its configuration is controlled by the
	// MTLTexture descriptor. Textures created with storageModeMemoryless don't have an IOAccelResource at any point in their
	// lifetime. The only way to populate such resource is to perform rendering operations on it. Blit operations are disallowed
	storageModeMemoryless storageMode = 3
)

// hazardTrackingMode Describes how hazard tracking is performed.
type hazardTrackingMode uint64

const (
	// hazardTrackingModeDefault The default hazard tracking mode for the context. Refer to the usage of the field for semantics.
	hazardTrackingModeDefault hazardTrackingMode = 0
	// hazardTrackingModeUntracked Do not perform hazard tracking.
	hazardTrackingModeUntracked hazardTrackingMode = 1
	// hazardTrackingModeTracked Do perform hazard tracking.
	hazardTrackingModeTracked hazardTrackingMode = 2
)

const (
	resourceCPUCacheModeShift = 0
	ResourceCPUCacheModeMask  = 0xf << resourceCPUCacheModeShift

	resourceStorageModeShift = 4
	ResourceStorageModeMask  = 0xf << resourceStorageModeShift

	resourceHazardTrackingModeShift = 8
	ResourceHazardTrackingModeMask  = 0x3 << resourceHazardTrackingModeShift
)

// resourceOptions A set of optional arguments to influence the creation of a MTLResource (MTLTexture or MTLBuffer.
type resourceOptions uint64

const (
	ResourceCPUCacheModeDefaultCache  = resourceOptions(cpuCacheModeDefaultCache << resourceCPUCacheModeShift)
	ResourceCPUCacheModeWriteCombined = resourceOptions(cpuCacheModeWriteCombined << resourceCPUCacheModeShift)

	ResourceStorageModeShared     = resourceOptions(storageModeShared << resourceStorageModeShift)
	ResourceStorageModeManaged    = resourceOptions(storageModeManaged << resourceStorageModeShift)
	ResourceStorageModePrivate    = resourceOptions(storageModePrivate << resourceStorageModeShift)
	ResourceStorageModeMemoryless = resourceOptions(storageModeMemoryless << resourceStorageModeShift)

	ResourceHazardTrackingModeDefault   = resourceOptions(hazardTrackingModeDefault << resourceHazardTrackingModeShift)
	ResourceHazardTrackingModeUntracked = resourceOptions(hazardTrackingModeUntracked << resourceHazardTrackingModeShift)
	ResourceHazardTrackingModeTracked   = resourceOptions(hazardTrackingModeTracked << resourceHazardTrackingModeShift)
)
