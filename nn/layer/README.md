# Layers Overview

This document explains the logic and data flow for each layer type in `nn/layer`.
All layers implement `Compile(device, input)` and return a `*num.Data` that holds
device buffers plus backprop hooks.

## Common Conventions
- Data layout is `MTLSize{W, H, D}`.
- `Compile` wires GPU ops and returns a new `*num.Data` that depends on input.
- Layers that own trainable weights implement `ForUpdate()` and `LoadFromProvider()`.
- Weight initialization uses the `initializer` type passed in; if it implements
  `DistributionProvider`, layers choose uniform vs normal accordingly.

## Layers

### Activation (ReLU)
File: `nn/layer/activation.go`
- `NewReLu()` wraps `device.Relu`.
- Forward: elementwise `max(0, x)`.
- Backward: passes gradient through when input > 0.

### Conv
File: `nn/layer/conv.go`
- Initializes weights and biases, then calls `device.Conv`.
- Weights shape: `filterSize x filterSize x (inputDepth * filtersCount)`.
- Biases shape: `1 x 1 x filtersCount`, one bias per filter.
- Uses `initWeights.GetNormK(fanIn, fanOut)` to scale initialization.
- Forward uses Metal kernel with padding/stride and supports batch in depth.
- Typical init: Kaiming (ReLU/SiLU) or Xavier (linear).
- Default init: `initializer.KaimingNormalReLU`.

### Dropout
File: `nn/layer/dropout.go`
- Wraps `device.Dropout`.
- Uses per-element random mask; output uses inverted dropout scaling so
  expected value matches input during training.
- Backward passes gradient only where mask kept the value, scaled identically.

### Embeddings
File: `nn/layer/embeddings.go`
- Wraps `device.Embeddings`.
- Input are token indices (floats with integer values).
- Outputs a matrix of token embeddings `[featuresCount x contextLength x batch]`.
- Backward accumulates into embedding table.

### Linear
File: `nn/layer/linear.go`
- Initializes weight matrix and optional bias.
- Forward: `input * weights` (matrix multiply).
- Optional bias: row-wise add with `device.AddRow`.
- Weights are returned via `ForUpdate()`.
- Typical init: Xavier (linear), Kaiming for ReLU/SiLU downstream.
- Default init: `initializer.XavierNormalLinear`.

### LinearWithWeights
File: `nn/layer/linear-with-weights.go`
- Uses externally provided weights.
- Forward: `input * weights`.
- No owned parameters.

### MulRows
File: `nn/layer/mulrows.go`
- Creates per-column weights (length = width).
- Forward: multiplies each row by weights (broadcast across rows).
- Intended for learnable per-feature scaling.
- Typical init: start from ones.

### Reshape
File: `nn/layer/reshape.go`
- Changes `MTLSize` metadata without copying buffers.
- Forward/backward are no-ops; gradients flow through unchanged.

### RMSNorm
File: `nn/layer/rmsnorm.go`
- Wraps `device.RMSNorm(input, input.Dims.W)`.
- Computes RMS per row, normalizes each element by its row RMS.
- No learned scale/bias in current implementation.

### Residual
File: `nn/layer/residual.go`
- Computes `input + Layers.Compile(input)`.
- Ensures skip connection with matching dimensions.

### SwiGLU
File: `nn/layer/swiglu.go`
- Two projections from input: `W1` and `W2`.
- Applies SiLU to `W1` output, then elementwise multiply with `W2` output.
- Final projection with `W3` back to `featuresCount`.
- Weights: `W1`, `W2` shaped `[hiddenSize x inputWidth]`, `W3` `[featuresCount x hiddenSize]`.
- Typical init: Kaiming for `W1/W2` (SiLU gating), Xavier or Kaiming for `W3`.
- Default init: `initializer.KaimingNormalReLU`.

### SAMultiHead (Self-Attention Multi-Head with RoPE)
File: `nn/layer/sa-miltihead.go`
- Creates Q, K, V linear projections from input.
- Applies RoPE to Q and K.
- Reshapes into heads, transposes for attention.
- Computes attention weights with scaled dot-product, applies triangular mask + softmax.
- Multiplies attention by V, then reshapes back to original feature layout.
- Typical init: Xavier or Kaiming (often Xavier in attention blocks).
- Default init: `initializer.XavierNormalLinear`.

## Layers Container
File: `nn/layer/layers.go`
- `Layers.Compile` chains layers sequentially.
- `ForUpdate` aggregates trainable parameters across layers.
- `LoadFromProvider` pulls external weights for all compatible layers.
