package vaephoto

import (
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/atkhx/metal/dataset"
)

const (
	ImageDepthRGB = 3
)

type PhotoDataset struct {
	images    []photoImage
	patchSize int
	rng       *rand.Rand
}

type photoImage struct {
	w, h int
	data []float32 // channel-first: R, G, B planes
}

func LoadPhotoDataset(dir string, patchSize int) (*PhotoDataset, error) {
	if patchSize < 1 {
		return nil, fmt.Errorf("patchSize must be >= 1")
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read dir: %w", err)
	}

	var images []photoImage
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(e.Name()))
		if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
			continue
		}
		fullPath := filepath.Join(dir, e.Name())
		img, err := decodeImage(fullPath)
		if err != nil {
			return nil, fmt.Errorf("decode %s: %w", fullPath, err)
		}
		if img.w < patchSize || img.h < patchSize {
			continue
		}
		images = append(images, img)
	}

	if len(images) == 0 {
		return nil, fmt.Errorf("no images loaded from %s", dir)
	}

	return &PhotoDataset{
		images:    images,
		patchSize: patchSize,
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}, nil
}

func (d *PhotoDataset) ReadRandomSampleBatch(batchSize int) (dataset.Sample, error) {
	if batchSize < 1 {
		return dataset.Sample{}, fmt.Errorf("batchSize must be >= 1")
	}

	patchWH := d.patchSize * d.patchSize
	out := make([]float32, batchSize*patchWH*ImageDepthRGB)

	for i := 0; i < batchSize; i++ {
		img := d.images[d.rng.Intn(len(d.images))]
		x0 := d.rng.Intn(img.w - d.patchSize + 1)
		y0 := d.rng.Intn(img.h - d.patchSize + 1)

		for c := 0; c < ImageDepthRGB; c++ {
			srcPlane := c * img.w * img.h
			dstPlane := i*patchWH*ImageDepthRGB + c*patchWH

			for y := 0; y < d.patchSize; y++ {
				srcRow := (y0+y)*img.w + x0
				dstRow := y * d.patchSize
				copy(out[dstPlane+dstRow:dstPlane+dstRow+d.patchSize], img.data[srcPlane+srcRow:srcPlane+srcRow+d.patchSize])
			}
		}
	}

	return dataset.Sample{Input: out}, nil
}

func (d *PhotoDataset) ReadRandomSampleBatchTo(batchSize int, out []float32) error {
	if batchSize < 1 {
		return fmt.Errorf("batchSize must be >= 1")
	}

	patchWH := d.patchSize * d.patchSize
	//out := make([]float32, batchSize*patchWH*ImageDepthRGB)

	for i := 0; i < batchSize; i++ {
		img := d.images[d.rng.Intn(len(d.images))]
		x0 := d.rng.Intn(img.w - d.patchSize + 1)
		y0 := d.rng.Intn(img.h - d.patchSize + 1)

		for c := 0; c < ImageDepthRGB; c++ {
			srcPlane := c * img.w * img.h
			dstPlane := i*patchWH*ImageDepthRGB + c*patchWH

			for y := 0; y < d.patchSize; y++ {
				srcRow := (y0+y)*img.w + x0
				dstRow := y * d.patchSize
				copy(
					out[dstPlane+dstRow:dstPlane+dstRow+d.patchSize],
					img.data[srcPlane+srcRow:srcPlane+srcRow+d.patchSize],
				)
			}
		}
	}
	return nil
}

func decodeImage(path string) (photoImage, error) {
	f, err := os.Open(path)
	if err != nil {
		return photoImage{}, err
	}
	defer f.Close()

	src, _, err := image.Decode(f)
	if err != nil {
		return photoImage{}, err
	}

	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	rgba := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.Draw(rgba, rgba.Bounds(), src, b.Min, draw.Src)

	wh := w * h
	data := make([]float32, wh*ImageDepthRGB)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := (y*rgba.Stride + x*4)

			r := float32(rgba.Pix[off]) / 255.0
			g := float32(rgba.Pix[off+1]) / 255.0
			bv := float32(rgba.Pix[off+2]) / 255.0

			idx := y*w + x
			data[idx] = r
			data[wh+idx] = g
			data[2*wh+idx] = bv
		}
	}

	return photoImage{w: w, h: h, data: data}, nil
}
