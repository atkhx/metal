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
`NewTransformerBlock(cfg, init)` builds two residual branches with configurable
RMSNorm, MulRows, and Dropout:
`Residual([RMSNorm? -> MulRows?] -> MHA -> Linear -> Dropout?)`
then
`Residual([RMSNorm? -> MulRows?] -> SwiGLU -> Dropout?)`.

## Transformer Stack
`NewTransformer(...)` stacks N transformer blocks. The `withHead` flag controls
whether it adds a final RMSNorm + MulRows + tied Linear head and reshapes output
for `CrossEntropyPos`.

Example:
```
// Encoder-only stack (no LM head)
layers := model.NewTransformer(ctx, dModel, heads, headSize, dFF, blocks, vocab, batch, drop, false, &initCfg, device, nil)
```

## Conv Feature Extractor
`NewConvFeatureExtractor(...)` builds a stack of `Conv -> ReLU` blocks.

## ConvNet
`NewConvNet(...)` builds `Conv -> ReLU` blocks and optionally adds a Linear head
when `headFeatures > 0`.

## Transformer LM
The LM path is built via `NewTransformer(...)` with `withHead=true`.
