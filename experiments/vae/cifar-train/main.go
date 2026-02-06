package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	cifar_10 "github.com/atkhx/metal/dataset/cifar-10"
	"github.com/atkhx/metal/experiments/vae/pkg"
	"github.com/atkhx/metal/mtl"
	"github.com/atkhx/metal/nn/proc"
)

var (
	miniBatchSize = pkg.CIFARBatchSize
	latentDim     = pkg.CIFARLatentDim
	epochs        = 20000
	statSize      = 100
	klBeta        = float32(latentDim) / float32(cifar_10.ImageSizeRGB)

	datasetPath = "./data/cifar-10"
	weightsPath = "./data/vae-cifar-10/"
	weightsFile = weightsPath + "model.json"
)

func main() {
	var err error
	defer func() {
		if err != nil {
			log.Fatalln(err)
		}
	}()

	if err = os.MkdirAll(weightsPath, os.ModePerm); err != nil {
		err = fmt.Errorf("failed to create weights path: %w", err)
		return
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	device := proc.NewWithSystemDefaultDevice()
	defer device.Release()

	optimizer := pkg.CreateOptimizer(epochs, device)

	vaeModel := pkg.CreateCifarVAETrainModel(miniBatchSize, latentDim, device, optimizer)
	vaeModel.Compile()

	if err = vaeModel.LoadFromFile(weightsFile); err != nil {
		return
	}

	defer func() {
		if err := vaeModel.SaveToFile(weightsFile); err != nil {
			log.Fatalln(err)
		}
	}()

	input, output := vaeModel.GetInput(), vaeModel.GetOutput()

	reconLoss := device.BinaryCrossEntropy(output, input)
	reconMean := device.Mean(reconLoss)

	muLogVar := vaeModel.GetMuLogVar()
	klLoss := device.VAEKLDivergence(muLogVar, latentDim)
	klMean := device.Mean(klLoss)

	klWeight := device.NewDataWithValues(klMean.Dims, []float32{klBeta})
	totalLoss := device.Add(reconMean, device.MulEqual(klMean, klWeight))
	pipeline := device.GetTrainingPipeline(totalLoss)

	trainDataset, err := cifar_10.CreateTrainingDataset(datasetPath)
	if err != nil {
		err = fmt.Errorf("cifar_10.CreateTrainingDataset: %w", err)
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

		batchInput, _ := trainDataset.ReadRandomSampleBatch(miniBatchSize)
		copy(input.Data.GetFloats(), batchInput.Input)
		pipeline.TrainIteration(func(b *mtl.CommandBuffer) {
			vaeModel.Update(b, iteration)
		})

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
