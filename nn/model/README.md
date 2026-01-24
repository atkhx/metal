# Model Recipes

This folder provides small factory helpers to assemble common layer stacks with
consistent initializer defaults.

## InitConfig
`InitConfig` lets you override distribution or per-component initializers.

If you pass `nil` to a recipe, defaults are used:
- Conv: Kaiming Normal (ReLU gain)
- Linear: Xavier Normal (linear gain)
- Attention: Xavier Normal (linear gain)
- FFN: Kaiming Normal (ReLU gain)

Use `DefaultInitConfigUniform()` or `DefaultInitConfigNormal()` to switch all
defaults in one place, then override specific fields if needed.

## Transformer Block
`NewTransformerBlock(...)` builds:
`Residual(RMSNorm -> MHA -> Dropout)` then `Residual(RMSNorm -> SwiGLU -> Dropout)`.

## Conv Feature Extractor
`NewConvFeatureExtractor(...)` builds a stack of `Conv -> ReLU` blocks.
