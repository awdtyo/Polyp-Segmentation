"""The assembled segmentation model and its parameter groups.

Milestone 4 assembles the dual-branch model: ConvNeXt (Branch A) and the
Meta-Polyp CAFormer encoder (Branch B) run on the same input, each stage is
aligned to the shared width, fused by concatenation or cross-attention, and the
plain decoder produces logits. The ``single_a`` and ``single_b`` presets keep
the Milestone 2-3 baselines and skip the unused encoder entirely.

MSU (Milestone 5) and the Convformer skip block remain unimplemented. Anything
the config asks for that is not implemented raises ``NotImplementedError`` with
the milestone that owns it, so an experiment can never silently train a
different architecture than it claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor, nn

from src.models.convnext_enc import ConvNeXtEncoder
from src.models.decoder import ChannelAlignment, PlainDecoder
from src.models.fusion import FusionStack, fusion_branch, resolve_fusion_modes
from src.models.metapolyp_enc import MetaPolypEncoder

#: Encoder choices accepted by ``model.encoder``. ``dual`` means both branches
#: run; the two single values match the corresponding fusion preset.
ENCODER_KINDS: frozenset[str] = frozenset({"convnext", "caformer", "dual"})


class PolypSegNet(nn.Module):
    """Dual- or single-branch segmentation net: encoders -> align -> fuse -> decoder.

    Args:
        encoder_a: Branch A encoder exposing ``forward -> list[Tensor]`` and
            ``.out_channels``. Required for ``single_a`` and ``dual``.
        channels: Fused width per stage ``C1..C4`` (stages 1-4). The alignment
            projects each active encoder's widths onto these.
        encoder_b: Branch B encoder. Required for ``single_b`` and ``dual``.
        fusion: Preset name or four per-stage modes. ``hybrid`` is
            ``[concat, concat, xattn, xattn]``; ``all_concat``, ``single_a`` and
            ``single_b`` are the other presets.
        fusion_heads: Attention heads for every ``xattn`` stage. The stage width
            must be divisible by this number.
        pos_enc_2d: Add a fixed 2-D sinusoidal encoding before cross-attention.

    Forward:
        ``x: (B, 3, H, W)`` -> logits ``(B, 1, H, W)``.

    Raises:
        ValueError: if the encoders do not match the fusion mode, or a width is
            not divisible by ``fusion_heads``.
    """

    def __init__(
        self,
        encoder_a: nn.Module | None,
        channels: Sequence[int] = (64, 128, 320, 512),
        encoder_b: nn.Module | None = None,
        fusion: str | Sequence[str] = "single_a",
        fusion_heads: int = 8,
        pos_enc_2d: bool = False,
    ) -> None:
        super().__init__()
        modes = resolve_fusion_modes(fusion)
        branch = fusion_branch(modes)

        self.fusion = modes
        self.branch = branch
        self.channels = [int(c) for c in channels]
        if len(self.channels) != 4:
            raise ValueError(f"channels must have 4 entries (stages 1-4), got {self.channels}")
        self.decoder = PlainDecoder(self.channels)

        if branch == "single_a":
            if encoder_a is None:
                raise ValueError("fusion=single_a requires encoder_a")
            if encoder_b is not None:
                raise ValueError("fusion=single_a skips Branch B; do not pass encoder_b")
            self.encoder_a: nn.Module | None = encoder_a
            self.encoder_b: nn.Module | None = None
            self.align_a: nn.ModuleList | None = nn.ModuleList(
                [
                    ChannelAlignment(int(c_in), c_out)
                    for c_in, c_out in zip(encoder_a.out_channels, self.channels)
                ]
            )
            self.align_b: nn.ModuleList | None = None
            self.fusion_stack: nn.Module | None = None
        elif branch == "single_b":
            if encoder_b is None:
                raise ValueError("fusion=single_b requires encoder_b")
            if encoder_a is not None:
                raise ValueError("fusion=single_b skips Branch A; do not pass encoder_a")
            self.encoder_a = None
            self.encoder_b = encoder_b
            self.align_a = None
            self.align_b = nn.ModuleList(
                [
                    ChannelAlignment(int(c_in), c_out)
                    for c_in, c_out in zip(encoder_b.out_channels, self.channels)
                ]
            )
            self.fusion_stack = None
        else:
            if encoder_a is None or encoder_b is None:
                raise ValueError("dual-branch fusion requires both encoder_a and encoder_b")
            self.encoder_a = encoder_a
            self.encoder_b = encoder_b
            self.align_a = nn.ModuleList(
                [
                    ChannelAlignment(int(c_in), c_out)
                    for c_in, c_out in zip(encoder_a.out_channels, self.channels)
                ]
            )
            self.align_b = nn.ModuleList(
                [
                    ChannelAlignment(int(c_in), c_out)
                    for c_in, c_out in zip(encoder_b.out_channels, self.channels)
                ]
            )
            self.fusion_stack = FusionStack(
                modes,
                self.channels,
                num_heads=fusion_heads,
                pos_enc_2d=pos_enc_2d,
            )

    @staticmethod
    def _encoder_maps(encoder: nn.Module, x: Tensor) -> list[Tensor]:
        """Run one encoder and enforce the four-map contract.

        Args:
            encoder: Encoder exposing ``forward -> list[Tensor]``.
            x: ``(B, 3, H, W)`` input.

        Returns:
            Four stage maps.

        Raises:
            ValueError: if the encoder returns a different number of maps.
        """
        maps = list(encoder(x))
        if len(maps) != 4:
            raise ValueError(f"Encoder returned {len(maps)} maps, expected 4")
        return maps

    @staticmethod
    def _align(alignments: nn.ModuleList, maps: Sequence[Tensor]) -> list[Tensor]:
        """Apply each stage's channel alignment.

        Args:
            alignments: Four ``ChannelAlignment`` modules.
            maps: Four encoder maps.

        Returns:
            Four aligned maps.
        """
        return [align(feature) for align, feature in zip(alignments, maps)]

    def forward(self, x: Tensor) -> Tensor:
        """Segment one batch.

        Args:
            x: ``(B, 3, H, W)`` normalised input, H and W divisible by 32.

        Returns:
            Logits ``(B, 1, H, W)``. Apply sigmoid outside the model.

        Raises:
            RuntimeError: if the module was assembled without an active branch.
        """
        if self.branch == "single_a":
            if self.encoder_a is None or self.align_a is None:
                raise RuntimeError("single_a model is missing encoder_a/align_a")
            fused = self._align(self.align_a, self._encoder_maps(self.encoder_a, x))
        elif self.branch == "single_b":
            if self.encoder_b is None or self.align_b is None:
                raise RuntimeError("single_b model is missing encoder_b/align_b")
            fused = self._align(self.align_b, self._encoder_maps(self.encoder_b, x))
        else:
            if (
                self.encoder_a is None
                or self.encoder_b is None
                or self.align_a is None
                or self.align_b is None
                or self.fusion_stack is None
            ):
                raise RuntimeError("dual-branch model is missing an encoder, alignment or fusion stack")
            maps_a = self._align(self.align_a, self._encoder_maps(self.encoder_a, x))
            maps_b = self._align(self.align_b, self._encoder_maps(self.encoder_b, x))
            fused = self.fusion_stack(maps_a, maps_b)
        return self.decoder(fused)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into the differential-learning-rate groups of section 4.

        Returns:
            Mapping with keys ``encoder_a``, ``encoder_b`` and ``head``. The
            inactive encoder group is empty; the caller drops empties. Every new
            module (alignment, fusion, decoder, head) is in ``head``. Every
            parameter of the model appears in exactly one group.
        """
        head: list[nn.Parameter] = list(self.decoder.parameters())
        for module in (self.align_a, self.align_b, self.fusion_stack):
            if module is not None:
                head += list(module.parameters())
        groups: dict[str, list[nn.Parameter]] = {
            "encoder_a": list(self.encoder_a.parameters()) if self.encoder_a is not None else [],
            "encoder_b": list(self.encoder_b.parameters()) if self.encoder_b is not None else [],
            "head": head,
        }
        return groups


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count parameters.

    Args:
        model: Any module.

    Returns:
        ``(total, trainable)`` parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(cfg: Mapping[str, Any]) -> PolypSegNet:
    """Build the model described by a resolved config.

    Args:
        cfg: Resolved config; reads ``model.encoder``, ``model.encoder_a``,
            ``model.encoder_b``, the pretrained flags, ``model.channels``,
            ``model.fusion``, ``model.fusion_heads``, ``model.pos_enc_2d``,
            ``model.msu``, ``model.msu_mode`` and ``model.skip_convformer``.

    Returns:
        The assembled model, on CPU, in train mode.

    Raises:
        ValueError: if ``model.encoder`` is unknown or does not match
            ``model.fusion``.
        NotImplementedError: if the config asks for a component that a later
            milestone owns (MSU, the Convformer skip block).
    """
    model_cfg = cfg["model"]
    encoder_kind = str(model_cfg["encoder"])
    if encoder_kind not in ENCODER_KINDS:
        raise ValueError(f"model.encoder must be one of {sorted(ENCODER_KINDS)}, got {encoder_kind!r}")

    if bool(model_cfg["skip_convformer"]):
        raise NotImplementedError(
            "model.skip_convformer=true is not implemented: the paper's Convformer skip block "
            "is deliberately out of scope (AGENTS.md section 11). Leave it false."
        )

    modes = resolve_fusion_modes(model_cfg["fusion"])
    if bool(model_cfg["msu"]):
        if model_cfg["msu_mode"] not in {"paper", "released"}:
            raise ValueError(f"Unknown msu_mode {model_cfg['msu_mode']!r}")
        raise NotImplementedError(
            "MSU is not implemented yet (Milestone 5); leave model.msu false for the baselines"
        )

    branch = fusion_branch(modes)
    expected_encoder = {"single_a": "convnext", "single_b": "caformer", "dual": "dual"}[branch]
    if encoder_kind != expected_encoder:
        raise ValueError(
            f"model.encoder={encoder_kind!r} is inconsistent with model.fusion={modes}; "
            f"use model.encoder={expected_encoder!r} (convnext for single_a, "
            "caformer for single_b, dual for mixed concat/xattn)"
        )

    if branch == "single_a":
        encoder_a = ConvNeXtEncoder(
            model_name=str(model_cfg["encoder_a"]),
            pretrained=bool(model_cfg["pretrained_a"]),
        )
        return PolypSegNet(
            encoder_a=encoder_a,
            channels=model_cfg["channels"],
            encoder_b=None,
            fusion=modes,
            fusion_heads=int(model_cfg["fusion_heads"]),
            pos_enc_2d=bool(model_cfg["pos_enc_2d"]),
        )

    if branch == "single_b":
        encoder_b = MetaPolypEncoder(
            model_name=str(model_cfg["encoder_b"]),
            pretrained=bool(model_cfg["pretrained_b"]),
        )
        return PolypSegNet(
            encoder_a=None,
            channels=model_cfg["channels"],
            encoder_b=encoder_b,
            fusion=modes,
            fusion_heads=int(model_cfg["fusion_heads"]),
            pos_enc_2d=bool(model_cfg["pos_enc_2d"]),
        )

    encoder_a = ConvNeXtEncoder(
        model_name=str(model_cfg["encoder_a"]),
        pretrained=bool(model_cfg["pretrained_a"]),
    )
    encoder_b = MetaPolypEncoder(
        model_name=str(model_cfg["encoder_b"]),
        pretrained=bool(model_cfg["pretrained_b"]),
    )
    return PolypSegNet(
        encoder_a=encoder_a,
        channels=model_cfg["channels"],
        encoder_b=encoder_b,
        fusion=modes,
        fusion_heads=int(model_cfg["fusion_heads"]),
        pos_enc_2d=bool(model_cfg["pos_enc_2d"]),
    )
