package cifar_100

import "errors"

const (
	ImageWidth  = 32
	ImageHeight = 32
	ImageDepth  = 3

	ClassSize = 1
	LabelSize = 1

	ClassesCount = 100

	ImageSizeGray = ImageWidth * ImageHeight
	ImageSizeRGB  = ImageWidth * ImageHeight * ImageDepth
	SampleSize    = ClassSize + LabelSize + ImageSizeRGB
)

var ErrorIndexOutOfRange = errors.New("index out of range")
