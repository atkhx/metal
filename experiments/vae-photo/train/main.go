package main

import (
	"context"
	"fmt"
	"log"
	"os/signal"
	"syscall"
	"time"

	vaephoto "github.com/atkhx/metal/experiments/vae-photo/pkg"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/proc"
)

var (
	miniBatchSize = vaephoto.PhotoBatchSize
	latentDim     = vaephoto.PhotoLatentDim
	epochs        = 5000
	statSize      = 100
	klBeta        = float32(latentDim) / float32(vaephoto.PhotoPatchSize*vaephoto.PhotoPatchSize*vaephoto.ImageDepthRGB)

	datasetPath = "./data/vae-photo/data"
	weightsFile = "./data/vae-photo/model.json"
)

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	optimizer := device.GetOptimizerAdam(epochs, 0.9, 0.99, 0.0003, 0.000000001)

	vaeModel := vaephoto.CreatePhotoVAETrainModel(miniBatchSize, latentDim, device, optimizer)
	vaeModel.Compile()

	if err = vaeModel.LoadFromFile(weightsFile); err != nil {
		return
	}

	input, output := vaeModel.GetInput(), vaeModel.GetOutput()

	fmt.Println("input:", input.Dims, "output:", output.Dims)
	defer func() {
		if err := vaeModel.SaveToFile(weightsFile); err != nil {
			log.Fatalln(err)
		}
	}()

	reconLoss := device.BinaryCrossEntropy(output, input)
	reconMean := device.Mean(reconLoss)

	muLogVar := vaeModel.GetMuLogVar()
	klLoss := device.VAEKLDivergence(muLogVar, latentDim)
	klMean := device.Mean(klLoss)

	klWeight := device.NewDataWithValues(klMean.Dims, []float32{klBeta})
	totalLoss := device.Add(reconMean, device.MulEqual(klMean, klWeight))
	pipeline := device.GetTrainingPipeline(totalLoss)

	trainDataset, err := vaephoto.LoadPhotoDataset(datasetPath, vaephoto.PhotoPatchSize)
	if err != nil {
		err = fmt.Errorf("LoadPhotoDataset: %w", err)
		return
	}

	var t = time.Now()
	var lossAvg float32
	for iteration := 0; iteration < epochs; iteration++ {
		select {
		case <-ctx.Done():
			return
		default:
		}

		//t0 := time.Now()
		//t1 := time.Now()
		trainDataset.ReadRandomSampleBatchTo(miniBatchSize, input.Data.GetFloats())
		//sampleDuration := time.Since(t1)
		//t1 = time.Now()
		//var trainDuration, updateDuration time.Duration
		pipeline.TrainIteration(func(b *mtl.CommandBuffer) {
			//trainDuration = time.Since(t1)
			//t1 = time.Now()
			vaeModel.Update(b, iteration)
			//updateDuration = time.Since(t1)
		})

		//totalDuration := time.Since(t0)
		//fmt.Println("sample:", sampleDuration, "\ttrain:", trainDuration, "\tupdate:", updateDuration, "\ttotal:", totalDuration)
		//fmt.Println("sample:", sampleDuration, "\ttotal:", totalDuration)

		// sample: _24.083µs       train: 267.125µs        update: 13.917µs        total: 49.560792ms
		// sample: 246.083µs       train: 323.833µs        update: 14.958µs        total: 444.254ms

		lossAvg += totalLoss.Data.GetFloats()[0]
		if (iteration > 0 || statSize == 1) && iteration%statSize == 0 {
			lossAvg /= float32(statSize)
			fmt.Println(
				fmt.Sprintf("lossFunc: %.8f", lossAvg), "\t",
				"iteration:", iteration, "\t",
				"duration:", time.Since(t), "\t",
			)
			lossAvg = 0
			t = time.Now()
		}
	}
}
