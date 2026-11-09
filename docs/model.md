# Model Documentation: exp006 (all_concat dual-encoder)

This document describes the exact architecture of the `exp006_all_concat` configuration, as implemented in the codebase at commit `9cc9886` (branch `feat/exp009-msu-pilots`). All details are derived from the source code and config files; nothing is inferred.

---

## 1. Encoders

Both encoders are built via `timm.create_model(..., features_only=True, out_indices=(0,1,2,3))` and return four feature maps at strides 4, 8, 16, 32.

### 1.1 Branch A — ConvNeXt (`src/models/convnext_enc.py:ConvNeXtEncoder`)

| Parameter | Value |
|-----------|-------|
| `model_name` | `convnext_base.fb_in1k` (pinned tag) |
| `pretrained` | `true` (config `model.pretrained_a`; tests/smoke runs use `false`) |
| `out_indices` | `(0, 1, 2, 3)` |
| Output strides | `[4, 8, 16, 32]` (checked at construction) |
| Output channels | `[128, 256, 512, 1024]` (from `feature_info.channels()`) |
| Stem | ConvNeXt stem: 4×4 conv, stride 4, 128 channels (part of stage 0 in timm) |
| Stages | 4 stages of ConvNeXt blocks (depths `[3, 3, 27, 3]` for base) |

**File/line references:**
- Class definition: `src/models/convnext_enc.py:21`
- `forward()` returns `list[self.backbone(x)]` (line 87)
- `out_channels` and `strides` from `feature_info` (lines 60–61)
- Stride/contract check: lines 63–67

### 1.2 Branch B — Meta-Polyp / CAFormer-S18 (`src/models/metapolyp_enc.py:MetaPolypEncoder`)

| Parameter | Value |
|-----------|-------|
| `model_name` | `caformer_s18.sail_in1k` (pinned tag) |
| `pretrained` | `true` (config `model.pretrained_b`; tests/smoke runs use `false`) |
| `out_indices` | `(0, 1, 2, 3)` |
| Output strides | `[4, 8, 16, 32]` (checked at construction) |
| Output channels | `[64, 128, 320, 512]` (constant `META_POLYP_CHANNELS`, line 22) |
| Stem | CAFormer stem (3×3 conv, stride 2 + 3×3 conv, stride 2 → stride 4 overall) |
| Stages | 4 stages: stages 1–2 use ConvFormer blocks (separable-conv token mixer), stages 3–4 use Transformer blocks (self-attention token mixer); depths `[3, 3, 9, 3]` |

**File/line references:**
- Class definition: `src/models/metapolyp_enc.py:25`
- `forward()` returns `list[self.backbone(x)]` (line 92)
- Width/stride check against `META_POLYP_CHANNELS` and `EXPECTED_STRIDES`: lines 67–72

**Note:** The Meta-Polyp paper's released code uses a CAFormer-S18 backbone. The timm model `caformer_s18.sail_in1k` matches the published depths and widths exactly. No TensorFlow→PyTorch weight remap is performed; both branches use timm ImageNet weights.

---

## 2. Channel Alignment (`src/models/decoder.py:ChannelAlignment`)

Before fusion, each branch's feature map at stage *i* is projected to the shared fused width `C_i` via a **1×1 convolution + BatchNorm** (no activation by default).

| Stage | Stride | ConvNeXt in-channels | CAFormer in-channels | Fused width `C_i` |
|-------|--------|---------------------|----------------------|-------------------|
| 1     | 4      | 128                 | 64                   | 64                |
| 2     | 8      | 256                 | 128                  | 128               |
| 3     | 16     | 512                 | 320                  | 320               |
| 4     | 32     | 1024                | 512                  | 512               |

**Module:** `ChannelAlignment(in_channels, out_channels, use_activation=False)` (lines 37–70)
- `conv`: `nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)`
- `norm`: `BatchNorm2d(out_channels)`
- `act`: `Identity` (since `use_activation=False`)

**File/line references:**
- `ChannelAlignment.__init__`: lines 55–59
- `ChannelAlignment.forward`: lines 61–70
- Instantiated in `PolypSegNet.__init__` for each branch: lines 88–93 (branch A), 116–126 (branch B)

---

## 3. Fusion: `model.fusion: all_concat` (`src/models/fusion.py`)

`all_concat` resolves to `["concat", "concat", "concat", "concat"]` via `FUSION_PRESETS` (line 45). Every stage uses `ConcatFusion`.

### 3.1 Per-stage operation (`ConcatFusion`, lines 190–226)

For each stage *i* with fused width `C_i`:

1. **Inputs:** Two aligned maps `first` (ConvNeXt) and `second` (CAFormer), each `(B, C_i, H_i, W_i)`.
2. **Concatenate:** `torch.cat([first, second], dim=1)` → `(B, 2·C_i, H_i, W_i)`.
3. **Project:** 3×3 convolution `Conv2d(2·C_i, C_i, kernel_size=3, padding=1, bias=False)`.
4. **Normalize:** `BatchNorm2d(C_i)`.
5. **Activate:** `ReLU(inplace=True)`.
6. **Output:** `(B, C_i, H_i, W_i)` — passed to the decoder as the fused feature for that stage.

### 3.2 Fusion parameters per stage

| Stage | `C_i` | ConcatFusion params |
|-------|-------|---------------------|
| 1     | 64    | 73,856  (0.074 M)   |
| 2     | 128   | 295,168 (0.295 M)   |
| 3     | 320   | 1,843,712 (1.844 M) |
| 4     | 512   | 4,719,744 (4.720 M) |
| **Total fusion** | | **6,932,480 (6.932 M)** |

**File/line references:**
- `FUSION_PRESETS["all_concat"]`: `fusion.py:45`
- `resolve_fusion_modes`: `fusion.py:52–90`
- `ConcatFusion.__init__`: `fusion.py:206–213`
- `ConcatFusion.forward`: `fusion.py:215–226`
- `FusionStack` builds four blocks: `fusion.py:343–351`
- `PolypSegNet.forward` calls `fusion_stack(maps_a, maps_b)`: `net.py:198`

### 3.3 What is passed to the decoder

The `FusionStack.forward` returns a list of four fused tensors `[F1, F2, F3, F4]` with shapes:
- `F1`: `(B, 64, H/4, W/4)`
- `F2`: `(B, 128, H/8, W/8)`
- `F3`: `(B, 320, H/16, W/16)`
- `F4`: `(B, 512, H/32, W/32)`

These become the `features` argument to `PlainDecoder.forward` (line 164 in `decoder.py`).

---

## 4. Decoder (`src/models/decoder.py:PlainDecoder`)

`decoder.msu: false` (config) → no MSU module is registered (`self.msu is None`, line 156). The decoder is exactly the Milestone 4 U-Net decoder.

### 4.1 Architecture

| Block | Operation | Input channels | Output channels |
|-------|-----------|----------------|-----------------|
| `dec4` | `ConvBlock(c4, c4)` on `F4` | 512 | 512 |
| `up4_to_3` | `ConvTranspose2d(c4, c3, 2, 2)` | 512 | 320 |
| `dec3` | `ConvBlock(c3+c3, c3)` on `cat(up4_to_3(d4), F3)` | 640 | 320 |
| `up3_to_2` | `ConvTranspose2d(c3, c2, 2, 2)` | 320 | 128 |
| `dec2` | `ConvBlock(c2+c2, c2)` on `cat(up3_to_2(d3), F2)` | 256 | 128 |
| `up2_to_1` | `ConvTranspose2d(c2, c1, 2, 2)` | 128 | 64 |
| `dec1` | `ConvBlock(c1+c1, c1)` on `cat(up2_to_1(d2), F1)` | 128 | 64 |
| `head_up` | `ConvTranspose2d(c1, head_channels, 4, 4)` | 64 | 32 |
| `head_norm` | `BatchNorm2d(32)` | 32 | 32 |
| `head_act` | `ReLU` | 32 | 32 |
| `head_conv` | `Conv2d(32, 1, 1)` | 32 | 1 |

### 4.2 `ConvBlock` (lines 73–101)

Two layers of: `Conv2d(3×3, padding=1, bias=False)` → `BatchNorm2d` → `ReLU(inplace=True)`. Spatial size is preserved.

### 4.3 Head

- Upsamples `d1` (stride 4) to full resolution via `ConvTranspose2d(kernel_size=4, stride=4)`.
- Projects to 1 channel (logits). **No sigmoid** — applied only in loss and inference.

### 4.4 Effect of `decoder.msu: false`

- `self.msu = None` (line 156)
- The two MSU links (D4→D2 and D3→D1) are **not** applied.
- State dict keys match exactly the Milestone 4 decoder (tested in `test_msu.py`).

**File/line references:**
- `PlainDecoder.__init__`: lines 121–162
- `PlainDecoder.forward`: lines 164–200
- `ConvBlock`: lines 73–101
- `ChannelAlignment`: lines 37–70
- `make_norm` (BatchNorm): lines 25–34

---

## 5. Parameter Counts (pretrained=false, CPU)

| Component | Parameters | Millions |
|-----------|------------|----------|
| Encoder A (ConvNeXt-Base) | 87,564,416 | 87.564 |
| Encoder B (CAFormer-S18) | 23,236,912 | 23.237 |
| Alignment A (4 × 1×1 conv + BN) | 731,136 | 0.731 |
| Alignment B (4 × 1×1 conv + BN) | 387,072 | 0.387 |
| Fusion (4 × ConcatFusion) | 6,932,480 | 6.932 |
| Decoder + Head | 8,925,825 | 8.926 |
| **Total** | **127,777,841** | **127.778** |

**Verification:**
```python
from src.utils import load_config
from src.models.net import build_model, count_parameters
cfg = load_config("configs/exp006_all_concat.yaml")
cfg["model"]["pretrained_a"] = False
cfg["model"]["pretrained_b"] = False
model = build_model(cfg)
total, _ = count_parameters(model)
assert total == 127_777_841
```

---

## 6. Full Forward Pass Summary

```
Input: (B, 3, H, W)  — H, W divisible by 32, default 256×256
       Normalised with ImageNet mean/std (shared by both branches)

Branch A (ConvNeXt): x → [A1, A2, A3, A4]  — channels [128, 256, 512, 1024]
Branch B (CAFormer): x → [B1, B2, B3, B4]  — channels [64, 128, 320, 512]

Channel Alignment (per stage):
  A1' = align_a[0](A1)  — (B, 64, H/4, W/4)
  B1' = align_b[0](B1)  — (B, 64, H/4, W/4)
  ... similarly for stages 2, 3, 4

Fusion (all_concat = concat at all 4 stages):
  F1 = ConcatFusion(64)(A1', B1')   — (B, 64, H/4, W/4)
  F2 = ConcatFusion(128)(A2', B2')  — (B, 128, H/8, W/8)
  F3 = ConcatFusion(320)(A3', B3')  — (B, 320, H/16, W/16)
  F4 = ConcatFusion(512)(A4', B4')  — (B, 512, H/32, W/32)

Decoder:
  d4 = ConvBlock(F4)
  d3 = ConvBlock(cat(Up(d4), F3))
  d2 = ConvBlock(cat(Up(d3), F2))
  d1 = ConvBlock(cat(Up(d2), F1))
  logits = Head(d1)  — (B, 1, H, W)

Output: logits (sigmoid applied externally in loss / inference)
```

---

## 7. Items Not Verified from Code Alone

The following require a Colab run with real checkpoints and data:

- Actual VRAM usage on T4 (16 GB) at batch size 8, image size 256.
- Real training/validation Dice and IoU on Kvasir-SEG.
- Zero-shot performance on CVC-ClinicDB and CVC-ColonDB.
- Whether the `convnext_base.fb_in1k` and `caformer_s18.sail_in1k` pretrained weights load without mismatch (timm version compatibility).
- Exact epoch time and peak GPU memory as recorded in `run_info.json`.

All of the above are **not** claimed in this document; they are measured by the user on Colab and recorded in `PROGRESS.md`.