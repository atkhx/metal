package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/atkhx/metal/dataset/mnist"
	"github.com/atkhx/metal/experiments/vae/pkg"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/proc"
)

var (
	miniBatchSize = 16
	latentDim     = pkg.MNISTLatentDim

	datasetPath = "./data/mnist"
	weightsPath = "./data/vae-mnist/"
	weightsFile = weightsPath + "model.json"
	outputDir   = weightsPath + "/output"
)

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	mode := flag.String("mode", "random", "random | interp | grid | encode | encode_interp | mix")
	batch := flag.Int("batch", miniBatchSize, "batch size for generation")
	sigma := flag.Float64("sigma", 1.0, "stddev for random latent sampling")
	seed := flag.Int64("seed", time.Now().UnixNano(), "rng seed")
	gridN := flag.Int("grid", 0, "if >0, override mode to grid and batch to grid*grid")
	normalize := flag.Bool("normalize", false, "normalize each output image by min/max")
	dsPath := flag.String("dataset", datasetPath, "path to MNIST dataset")
	useTest := flag.Bool("test", false, "use test split for encode mode")
	flag.Parse()

	if *gridN > 0 {
		miniBatchSize = (*gridN) * (*gridN)
		*mode = "grid"
	} else {
		miniBatchSize = *batch
	}

	rand.Seed(*seed)

	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	var (
		vaeModel     *pkg.MnistVAE
		decoderModel *pkg.MnistVAE
	)

	switch *mode {
	case "encode":
		vaeModel = pkg.CreateMnistVAETrainModel(miniBatchSize, latentDim, device, nil)
		vaeModel.Compile()
	case "encode_interp", "mix":
		// Encoder with batch=2 (two source images), decoder with batch=miniBatchSize.
		vaeModel = pkg.CreateMnistVAETrainModel(2, latentDim, device, nil)
		decoderModel = pkg.CreateMnistVAEInferenceModel(miniBatchSize, latentDim, device, nil)
		vaeModel.Compile()
		decoderModel.Compile()
	default:
		vaeModel = pkg.CreateMnistVAEInferenceModel(miniBatchSize, latentDim, device, nil)
		vaeModel.Compile()
	}

	if err = vaeModel.LoadFromFile(weightsFile); err != nil {
		return
	}
	if decoderModel != nil {
		if err = decoderModel.LoadFromFile(weightsFile); err != nil {
			return
		}
	}

	outData := vaeModel.GetOutput()
	pipeline := device.GetInferencePipeline(outData)

	if err := os.MkdirAll(outputDir, os.ModePerm); err != nil {
		err = fmt.Errorf("os.MkdirAll: %w", err)
		return
	}

	input := vaeModel.GetInput().Data.GetFloats()
	switch *mode {
	case "random":
		fillLatentNormal(input, latentDim, miniBatchSize, float32(*sigma))
	case "interp":
		fillLatentInterp(input, latentDim, miniBatchSize, float32(*sigma))
	case "grid":
		fillLatentGrid(input, latentDim, *gridN, float32(*sigma))
	case "encode":
		var ds *mnist.Dataset
		if *useTest {
			ds, err = mnist.CreateTestingDataset(*dsPath)
		} else {
			ds, err = mnist.CreateTrainingDataset(*dsPath)
		}
		if err != nil {
			err = fmt.Errorf("mnist dataset: %w", err)
			return
		}
		batchInput, e := ds.ReadRandomSampleBatch(miniBatchSize)
		if e != nil {
			err = fmt.Errorf("ReadRandomSampleBatch: %w", e)
			return
		}
		copy(input, batchInput.Input)
	case "encode_interp", "mix":
		var ds *mnist.Dataset
		if *useTest {
			ds, err = mnist.CreateTestingDataset(*dsPath)
		} else {
			ds, err = mnist.CreateTrainingDataset(*dsPath)
		}
		if err != nil {
			err = fmt.Errorf("mnist dataset: %w", err)
			return
		}
		batchInput, e := ds.ReadRandomSampleBatch(2)
		if e != nil {
			err = fmt.Errorf("ReadRandomSampleBatch: %w", e)
			return
		}
		copy(input, batchInput.Input)
	default:
		err = fmt.Errorf("unknown mode: %s", *mode)
		return
	}

	start := time.Now()
	if *mode == "encode_interp" || *mode == "mix" {
		mu, e := encodeMu(vaeModel, latentDim, 2, device)
		if e != nil {
			err = e
			return
		}

		z := make([]float32, latentDim*miniBatchSize)
		switch *mode {
		case "encode_interp":
			fillLatentFromInterp(z, mu, latentDim, miniBatchSize)
		case "mix":
			fillLatentFromMix(z, mu, latentDim, miniBatchSize)
		}

		decInput := decoderModel.GetInput().Data.GetFloats()
		copy(decInput, z)
		outData = decoderModel.GetOutput()
		pipeline = device.GetInferencePipeline(outData)
	}

	pipeline.Forward()

	dims := mtl.MTLSize{W: mnist.ImageWidth, H: mnist.ImageHeight, D: mnist.ImageDepth * miniBatchSize}

	outFloats := outData.Data.GetFloats()
	outImgs, e := createGreyscaleImage(dims, outFloats[:dims.Length()], *normalize)
	if e != nil {
		err = fmt.Errorf("createGreyscaleImage output: %w", e)
		return
	}

	if *mode == "encode" {
		inImgs, e := createGreyscaleImage(dims, input[:dims.Length()], false)
		if e != nil {
			err = fmt.Errorf("createGreyscaleImage input: %w", e)
			return
		}
		for i := 0; i < miniBatchSize; i++ {
			inName := fmt.Sprintf("%s/input_%02d.png", outputDir, i)
			if e := os.WriteFile(inName, inImgs[i], os.ModePerm); e != nil {
				err = fmt.Errorf("os.WriteFile input: %w", e)
				return
			}
		}
	}

	if *mode == "encode_interp" || *mode == "mix" {
		inDims := mtl.MTLSize{W: mnist.ImageWidth, H: mnist.ImageHeight, D: 2}
		inImgs, e := createGreyscaleImage(inDims, input[:inDims.Length()], false)
		if e != nil {
			err = fmt.Errorf("createGreyscaleImage input: %w", e)
			return
		}
		for i := 0; i < 2; i++ {
			inName := fmt.Sprintf("%s/input_%02d.png", outputDir, i)
			if e := os.WriteFile(inName, inImgs[i], os.ModePerm); e != nil {
				err = fmt.Errorf("os.WriteFile input: %w", e)
				return
			}
		}
	}

	for i := 0; i < miniBatchSize; i++ {
		outName := fmt.Sprintf("%s/%s_%02d.png", outputDir, *mode, i)
		if e := os.WriteFile(outName, outImgs[i], os.ModePerm); e != nil {
			err = fmt.Errorf("os.WriteFile output: %w", e)
			return
		}
	}

	fmt.Println("generate done, mode:", *mode, "batch:", miniBatchSize, "duration:", time.Since(start))
}

func GetMinMaxValues[T float32 | float64](data []T) (min, max T) {
	for i := 0; i < len(data); i++ {
		if i == 0 || min > data[i] {
			min = data[i]
		}
		if i == 0 || max < data[i] {
			max = data[i]
		}
	}
	return
}

func createGreyscaleImage(dims mtl.MTLSize, data []float32, normalize bool) ([][]byte, error) {
	mh := dims.W
	mw := dims.H
	res := [][]byte{}

	wh := dims.W * dims.H

	for imageIndex := 0; imageIndex < dims.D; imageIndex++ {
		imgFloats := data[imageIndex*wh : (imageIndex+1)*wh]

		img := image.NewGray(image.Rect(0, 0, mw, mh))
		for y := 0; y < mh; y++ {
			for x := 0; x < mw; x++ {
				c := imgFloats[y*mw+x]
				if normalize {
					min, max := GetMinMaxValues(imgFloats)
					max -= min
					c = (c - min) / max
				}
				if c < 0 {
					c = 0
				}
				if c > 1 {
					c = 1
				}
				img.Set(x, y, color.Gray{Y: byte(255.0 * c)})
			}
		}

		buf := bytes.NewBuffer(nil)
		err := png.Encode(buf, img)
		if err != nil {
			return nil, err
		}

		res = append(res, buf.Bytes())
	}
	return res, nil
}

func fillLatentNormal(dst []float32, latentDim, batch int, sigma float32) {
	n := latentDim * batch
	for i := 0; i < n; i++ {
		dst[i] = float32(rand.NormFloat64()) * sigma
	}
}

func fillLatentInterp(dst []float32, latentDim, batch int, sigma float32) {
	if batch < 2 {
		fillLatentNormal(dst, latentDim, batch, sigma)
		return
	}
	z0 := make([]float32, latentDim)
	z1 := make([]float32, latentDim)
	for i := 0; i < latentDim; i++ {
		z0[i] = float32(rand.NormFloat64()) * sigma
		z1[i] = float32(rand.NormFloat64()) * sigma
	}
	for b := 0; b < batch; b++ {
		t := float32(b) / float32(batch-1)
		for j := 0; j < latentDim; j++ {
			dst[b*latentDim+j] = z0[j]*(1-t) + z1[j]*t
		}
	}
}

func fillLatentGrid(dst []float32, latentDim, gridN int, sigma float32) {
	if gridN < 1 {
		return
	}
	fillLatentNormal(dst, latentDim, gridN*gridN, sigma)
	if latentDim < 2 {
		return
	}
	for y := 0; y < gridN; y++ {
		for x := 0; x < gridN; x++ {
			b := y*gridN + x
			tx := float32(x)/float32(gridN-1)*2 - 1
			ty := float32(y)/float32(gridN-1)*2 - 1
			dst[b*latentDim+0] = tx * sigma
			dst[b*latentDim+1] = ty * sigma
		}
	}
}

func encodeMu(model *pkg.MnistVAE, latentDim, batch int, device *proc.Device) ([]float32, error) {
	muLogVar := model.GetMuLogVar()
	if muLogVar == nil {
		return nil, fmt.Errorf("muLogVar is nil")
	}
	encPipe := device.GetInferencePipeline(muLogVar)
	encPipe.Forward()

	raw := muLogVar.Data.GetFloats()
	out := make([]float32, latentDim*batch)
	for b := 0; b < batch; b++ {
		base := b * latentDim * 2
		copy(out[b*latentDim:(b+1)*latentDim], raw[base:base+latentDim])
	}
	return out, nil
}

func fillLatentFromInterp(dst, mu []float32, latentDim, batch int) {
	if batch < 2 {
		copy(dst, mu[:latentDim])
		return
	}
	z0 := mu[:latentDim]
	z1 := mu[latentDim : latentDim*2]
	for b := 0; b < batch; b++ {
		t := float32(b) / float32(batch-1)
		for j := 0; j < latentDim; j++ {
			dst[b*latentDim+j] = z0[j]*(1-t) + z1[j]*t
		}
	}
}

func fillLatentFromMix(dst, mu []float32, latentDim, batch int) {
	z0 := mu[:latentDim]
	z1 := mu[latentDim : latentDim*2]
	for b := 0; b < batch; b++ {
		for j := 0; j < latentDim; j++ {
			if rand.Intn(2) == 0 {
				dst[b*latentDim+j] = z0[j]
			} else {
				dst[b*latentDim+j] = z1[j]
			}
		}
	}
}
