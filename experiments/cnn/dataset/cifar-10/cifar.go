package cifar_10

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"strings"

	"github.com/atkhx/metal/experiments/cnn/dataset"
)

var ErrorIndexOutOfRange = errors.New("index out of range")

const (
	ImageWidth    = 32
	ImageHeight   = 32
	ImageDepthRGB = 3

	ImageSizeGray = ImageWidth * ImageHeight
	ImageSizeRGB  = ImageWidth * ImageHeight * 3
	SampleSize    = ImageSizeRGB + 1

	ClassesCount = 10
)

var cifarLabels = []string{
	"plane",
	"auto",
	"bird",
	"cat",
	"deer",
	"dog",
	"frog",
	"horse",
	"ship",
	"truck",
}

const (
	TrainImagesFileName = "cifar10-train-data.bin"
	TestImagesFileName  = "cifar10-test-data.bin"
)

func CreateTrainingDataset(datasetPath string) (*Dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, fmt.Errorf("can't open cifar-10 training file: %w", err)
	}
	return result, nil
}
func CreateTrainingDatasetGrayscale(datasetPath string) (*Dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TrainImagesFileName)

	result, err := Open(imagesFileName, false)
	if err != nil {
		return nil, fmt.Errorf("can't open cifar-10 training file: %w", err)
	}
	return result, nil
}

func CreateTestingDataset(datasetPath string) (*Dataset, error) {
	imagesFileName := fmt.Sprintf("%s/%s", strings.TrimRight(datasetPath, " /"), TestImagesFileName)

	result, err := Open(imagesFileName, true)
	if err != nil {
		return nil, fmt.Errorf("can't open cifar-10 testing file: %w", err)
	}
	return result, nil
}

func Open(filename string, rgb bool) (*Dataset, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	imagesCount := len(b) / SampleSize

	var images []float32

	//nolint:gomnd
	if rgb {
		images = make([]float32, imagesCount*ImageSizeRGB)
		for i := 0; i < imagesCount; i++ {
			imageOffset := i * ImageSizeRGB
			sampleOffset := i * SampleSize
			for j := 0; j < ImageSizeRGB; j++ {
				images[imageOffset+j] = float32(b[sampleOffset+1+j]) / 255.0
			}
		}
	} else {
		images = make([]float32, imagesCount*ImageSizeGray)
		for i := 0; i < imagesCount; i++ {
			imageOffset := i * ImageSizeGray
			sampleOffset := i * SampleSize

			for j := 0; j < ImageSizeGray; j++ {
				R := float32(b[sampleOffset+1+j]) / 255.0
				G := float32(b[sampleOffset+1+j+ImageWidth]) / 255.0
				B := float32(b[sampleOffset+1+j+ImageWidth+ImageHeight]) / 255.0

				images[imageOffset+j] = (R + G + B) / 3
			}
		}
	}

	var labelsIdx = make([]byte, imagesCount)
	for i := 0; i < imagesCount; i++ {
		labelsIdx[i] = b[i*SampleSize]
	}

	d := &Dataset{
		labelsIdx:   labelsIdx,
		images:      images,
		imagesCount: imagesCount,

		labels: cifarLabels,

		rgb: rgb,
	}
	return d, nil
}

type Dataset struct {
	labels []string
	images []float32

	imagesCount int
	labelsIdx   []byte

	rgb bool
}

func (d *Dataset) GetSamplesCount() int {
	return d.imagesCount
}

func (d *Dataset) GetClasses() []string {
	return d.labels
}

func (d *Dataset) ReadSample(index int) (dataset.Sample, error) {
	var imageSize = ImageSizeGray
	if d.rgb {
		imageSize = ImageSizeRGB
	}
	images := make([]float32, 0, imageSize)
	images = append(images, d.images[index*imageSize:(index+1)*imageSize]...)

	return dataset.Sample{
		Input:  images,
		Target: []float32{float32(d.labelsIdx[index])},
	}, nil
}

func (d *Dataset) ReadRandomSampleBatch(batchSize int) (dataset.Sample, error) {
	var imageSize = ImageSizeGray
	if d.rgb {
		imageSize = ImageSizeRGB
	}
	images := make([]float32, 0, batchSize*imageSize)
	labels := make([]float32, 0, batchSize)

	for i := 0; i < batchSize; i++ {
		index := rand.Intn(d.GetSamplesCount()) //nolint:gosec

		images = append(images, d.images[index*imageSize:(index+1)*imageSize]...)
		labels = append(labels, float32(d.labelsIdx[index]))
	}
	return dataset.Sample{
		Input:  images,
		Target: labels,
	}, nil
}
