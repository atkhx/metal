# Initializers

This package provides scale calculators and distribution hints for weight initialization.
Each initializer returns a scale `k` via `GetNormK(fanIn, fanOut)`. Layers then choose
uniform or normal sampling based on `DistributionProvider`:

- `DistributionUniform`: U(-k, k)
- `DistributionNormal`: N(0, k^2)

## Xavier (Glorot)
- **XavierUniform**: `k = gain * sqrt(6 / (fanIn + fanOut))`, U(-k, k)
- **XavierNormal**: `k = gain * sqrt(2 / (fanIn + fanOut))`, N(0, k^2)

Use Xavier for linear layers or attention projections when the activation is roughly linear.

## Kaiming (He)
- **KaimingUniform**: `k = gain * sqrt(6 / fanIn)`, U(-k, k)
- **KaimingNormal**: `k = gain * sqrt(2 / fanIn)`, N(0, k^2)

Use Kaiming for ReLU/SiLU-style activations.

## Fixed
`InitWeightFixed` returns a constant `k` and defaults to uniform sampling.

## Distribution Selection
If you pass an initializer implementing `DistributionProvider`, layers will pick
the correct sampling function. If you pass `nil`, layers use a reasonable default:

- Conv: `KaimingUniformReLU`
- Linear: `XavierUniformLinear`
- SwiGLU: `KaimingUniformReLU`
- SAMultiHead: `XavierUniformLinear`

You can override defaults by passing the desired initializer explicitly.
