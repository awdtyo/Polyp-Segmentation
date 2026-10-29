"""Per-stage fusion of the two encoder branches (AGENTS.md section 3.3).

Both branches produce four maps at strides 4, 8, 16 and 32. After channel
alignment, each stage is fused in one of two ways:

* ``concat`` (stages 1-2 by default): concatenate the two maps on the channel
  axis and project back to ``C_i`` with a 3x3 conv + norm + activation.
* ``xattn`` (stages 3-4 by default): flatten both maps to tokens and run
  cross-attention with the ConvNeXt tokens as query and the Meta-Polyp tokens
  as key/value. The attention output is added back to the query tokens, layer
  normalised, and reshaped to a feature map.

The stage-1 and stage-2 maps have 4096 and 1024 tokens at 256x256, versus 256
and 64 at stages 3 and 4. That is why only the deeper stages use attention.

``resolve_fusion_modes`` accepts either an explicit four-entry list or one of
the presets ``hybrid``, ``all_concat``, ``single_a`` and ``single_b``. The
single-branch presets are handled by the model assembly, which skips the unused
encoder entirely; :class:`FusionStack` itself only accepts the two dual-branch
modes.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn

#: Fusion modes that use both branches.
DUAL_FUSION_MODES: frozenset[str] = frozenset({"concat", "xattn"})
#: Fusion modes that use exactly one branch.
SINGLE_FUSION_MODES: frozenset[str] = frozenset({"single_a", "single_b"})
#: Every mode accepted in a per-stage fusion list.
STAGE_FUSION_MODES: frozenset[str] = DUAL_FUSION_MODES | SINGLE_FUSION_MODES

#: Named presets, stage 1 to stage 4.
FUSION_PRESETS: dict[str, tuple[str, str, str, str]] = {
    "hybrid": ("concat", "concat", "xattn", "xattn"),
    "all_concat": ("concat", "concat", "concat", "concat"),
    "single_a": ("single_a", "single_a", "single_a", "single_a"),
    "single_b": ("single_b", "single_b", "single_b", "single_b"),
}


def resolve_fusion_modes(fusion: str | Sequence[str]) -> list[str]:
    """Normalise ``model.fusion`` to exactly one mode per stage.

    Args:
        fusion: A preset name (``hybrid``, ``all_concat``, ``single_a``,
            ``single_b``), a dual mode repeated at every stage (``concat`` or
            ``xattn``), or a four-entry sequence of stage modes.

    Returns:
        List of four modes, stage 1 to stage 4.

    Raises:
        ValueError: if the value is not a known preset, a known repeated mode,
            or a four-entry list. A list may not mix single-branch and
            dual-branch modes, and may not mix ``single_a`` with ``single_b``.
    """
    if isinstance(fusion, str):
        if fusion in FUSION_PRESETS:
            return list(FUSION_PRESETS[fusion])
        if fusion in DUAL_FUSION_MODES:
            return [fusion] * 4
        raise ValueError(
            f"Unknown fusion setting {fusion!r}; use a preset {sorted(FUSION_PRESETS)}, "
            f"a repeated mode {sorted(DUAL_FUSION_MODES)}, or a four-entry list"
        )

    modes = list(fusion)
    if len(modes) != 4:
        raise ValueError(f"fusion must be a string or 4 modes, got {modes}")
    for mode in modes:
        if mode not in STAGE_FUSION_MODES:
            raise ValueError(f"Unknown fusion mode {mode!r}; expected one of {sorted(STAGE_FUSION_MODES)}")
    if any(mode in SINGLE_FUSION_MODES for mode in modes):
        if not all(mode in SINGLE_FUSION_MODES for mode in modes):
            raise ValueError(f"fusion cannot mix single-branch and dual-branch modes, got {modes}")
        if len(set(modes)) != 1:
            raise ValueError(f"fusion cannot mix single_a and single_b, got {modes}")
    return modes


def fusion_branch(modes: Sequence[str]) -> str:
    """Return which encoder branches a resolved fusion list needs.

    Args:
        modes: Four resolved stage modes.

    Returns:
        ``"single_a"``, ``"single_b"`` or ``"dual"``.

    Raises:
        ValueError: if the list does not contain four modes or mixes branch modes.
    """
    resolved = list(modes)
    if len(resolved) != 4:
        raise ValueError(f"Expected 4 fusion modes, got {len(resolved)}")
    if all(mode == "single_a" for mode in resolved):
        return "single_a"
    if all(mode == "single_b" for mode in resolved):
        return "single_b"
    if all(mode in DUAL_FUSION_MODES for mode in resolved):
        return "dual"
    raise ValueError(f"fusion list does not select a consistent branch layout: {resolved}")


def build_2d_sinusoidal_encoding(
    height: int,
    width: int,
    channels: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Build a 2D sinusoidal positional encoding for a flattened feature map.

    The encoding follows the usual transformer formulation: the token index in
    row-major order is the position, and sine/cosine pairs fill the channel
    dimension. It has no learnable parameters, so enabling it does not change
    the parameter count.

    Args:
        height: Feature-map height.
        width: Feature-map width.
        channels: Token width, expected to equal the fusion width ``C_i``.
        device: Optional tensor device.
        dtype: Optional output dtype (defaults to float32).

    Returns:
        ``(1, height * width, channels)`` float tensor. The leading batch axis
        broadcasts over the batch.

    Raises:
        ValueError: if any size is non-positive.
    """
    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError(f"height, width and channels must be positive, got {height}, {width}, {channels}")

    position = torch.arange(height * width, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, channels, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / float(channels))
    )
    encoding = torch.zeros(height * width, channels, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div_term)
    # For odd channel counts the cosine half is one column shorter.
    encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    encoding = encoding.unsqueeze(0)
    return encoding if dtype is None else encoding.to(dtype=dtype)


def _check_map_pair(first: Tensor, second: Tensor, channels: int, stage: int | None = None) -> None:
    """Validate a pair of aligned feature maps before fusing them.

    Args:
        first: ``(B, C, H, W)`` first branch map.
        second: ``(B, C, H, W)`` second branch map.
        channels: Expected ``C``.
        stage: Optional stage number for a clearer error message.

    Raises:
        ValueError: if either map is not 4D, has the wrong width, or the two
            maps have different spatial sizes.
    """
    label = "" if stage is None else f" at stage {stage}"
    if first.ndim != 4 or second.ndim != 4:
        raise ValueError(f"Fusion expects two 4D maps{label}, got {tuple(first.shape)} and {tuple(second.shape)}")
    if first.shape[1] != channels or second.shape[1] != channels:
        raise ValueError(
            f"Fusion expects {channels} channels per branch{label}, "
            f"got {first.shape[1]} and {second.shape[1]}"
        )
    if first.shape[-2:] != second.shape[-2:]:
        raise ValueError(
            f"Fusion branches must have the same spatial size{label}, "
            f"got {tuple(first.shape[-2:])} and {tuple(second.shape[-2:])}"
        )


class ConcatFusion(nn.Module):
    """Concatenate two aligned maps and project back to one stage width.

    Args:
        channels: Stage width ``C_i``. Both inputs have this width, the output
            also has this width.

    Attributes:
        conv: 3x3 convolution from ``2 * C_i`` to ``C_i``.
        norm: BatchNorm over ``C_i``.
        act: ReLU activation.

    Forward:
        ``(a, b)`` each ``(B, C_i, H, W)`` -> ``(B, C_i, H, W)``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = int(channels)
        self.conv = nn.Conv2d(2 * self.channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(self.channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        """Fuse by concatenation.

        Args:
            first: ``(B, C, H, W)`` aligned ConvNeXt map.
            second: ``(B, C, H, W)`` aligned Meta-Polyp map.

        Returns:
            ``(B, C, H, W)`` fused map.
        """
        _check_map_pair(first, second, self.channels)
        return self.act(self.norm(self.conv(torch.cat([first, second], dim=1))))


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion with ConvNeXt as query and Meta-Polyp as key/value.

    Args:
        channels: Stage width ``C_i``.
        num_heads: Attention heads; must divide ``channels``.
        pos_enc_2d: Add a fixed 2D sinusoidal encoding to both token streams
            before attention. Off by default, per AGENTS.md section 3.3.

    Attributes:
        attn: ``nn.MultiheadAttention`` with ``batch_first=True``.
        norm: LayerNorm applied after the residual add.

    Forward:
        ``(query, key_value)`` each ``(B, C_i, H, W)`` ->
        ``(B, C_i, H, W)``. The attention output is added to the query tokens
        before LayerNorm; no sigmoid or softmax is exposed.
    """

    def __init__(self, channels: int, num_heads: int = 8, pos_enc_2d: bool = False) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by fusion heads ({num_heads})"
            )
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.pos_enc_2d = bool(pos_enc_2d)
        self.attn = nn.MultiheadAttention(self.channels, self.num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.channels)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """Fuse two maps with cross-attention.

        Args:
            query: ``(B, C, H, W)`` aligned ConvNeXt map.
            key_value: ``(B, C, H, W)`` aligned Meta-Polyp map.

        Returns:
            ``(B, C, H, W)`` fused map.

        Raises:
            ValueError: if the two maps are incompatible.
        """
        _check_map_pair(query, key_value, self.channels)
        height, width = query.shape[-2], query.shape[-1]
        query_tokens = query.flatten(2).transpose(1, 2)  # (B, H*W, C)
        kv_tokens = key_value.flatten(2).transpose(1, 2)  # (B, H*W, C)

        if self.pos_enc_2d:
            encoding = build_2d_sinusoidal_encoding(
                height, width, self.channels, device=query.device, dtype=query.dtype
            )
            query_tokens = query_tokens + encoding
            kv_tokens = kv_tokens + encoding

        attended, _ = self.attn(
            query=query_tokens,
            key=kv_tokens,
            value=kv_tokens,
            need_weights=False,
        )
        fused = self.norm(query_tokens + attended)  # (B, H*W, C)
        return fused.transpose(1, 2).reshape(query.shape[0], self.channels, height, width).contiguous()


class FusionStack(nn.Module):
    """Apply one fusion block per stage to two lists of aligned maps.

    Args:
        modes: Four per-stage modes, each ``concat`` or ``xattn``. Use
            :func:`resolve_fusion_modes` first when reading a config.
        channels: Stage widths ``C1..C4``.
        num_heads: Attention heads for every ``xattn`` stage.
        pos_enc_2d: Enable the fixed 2D sinusoidal encoding in ``xattn`` stages.

    Forward:
        ``(maps_a, maps_b)``, each a list of four ``(B, C_i, H_i, W_i)``
        tensors -> list of four fused tensors with the same shapes.
    """

    def __init__(
        self,
        modes: Sequence[str],
        channels: Sequence[int],
        num_heads: int = 8,
        pos_enc_2d: bool = False,
    ) -> None:
        super().__init__()
        stage_modes = list(modes)
        widths = [int(c) for c in channels]
        if len(stage_modes) != 4:
            raise ValueError(f"modes must have 4 entries, got {len(stage_modes)}")
        if len(widths) != 4:
            raise ValueError(f"channels must have 4 entries, got {len(widths)}")
        unsupported = sorted(set(stage_modes) - DUAL_FUSION_MODES)
        if unsupported:
            raise ValueError(
                f"FusionStack only accepts dual-branch modes {sorted(DUAL_FUSION_MODES)}, "
                f"got {unsupported}; single-branch presets skip the unused encoder instead"
            )

        self.modes = stage_modes
        self.channels = widths
        self.blocks = nn.ModuleList(
            [
                ConcatFusion(width) if mode == "concat" else CrossAttentionFusion(width, num_heads, pos_enc_2d)
                for mode, width in zip(stage_modes, widths)
            ]
        )

    def forward(self, maps_a: Sequence[Tensor], maps_b: Sequence[Tensor]) -> list[Tensor]:
        """Fuse four aligned maps from each branch.

        Args:
            maps_a: Four ConvNeXt maps, stage 1 to stage 4.
            maps_b: Four Meta-Polyp maps, stage 1 to stage 4.

        Returns:
            List of four fused maps with the same shapes as the inputs.

        Raises:
            ValueError: if either list is not four maps or a stage pair is
                incompatible.
        """
        if len(maps_a) != 4 or len(maps_b) != 4:
            raise ValueError(f"Expected 4 maps per branch, got {len(maps_a)} and {len(maps_b)}")
        fused: list[Tensor] = []
        for stage, (block, first, second, channels) in enumerate(
            zip(self.blocks, maps_a, maps_b, self.channels), start=1
        ):
            _check_map_pair(first, second, channels, stage=stage)
            fused.append(block(first, second))
        return fused
