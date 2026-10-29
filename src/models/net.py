"""The assembled segmentation model and its parameter groups.

Milestone 2 builds the **baseline**: Branch A (ConvNeXt) -> channel alignment ->
plain decoder, no fusion, no MSU. Later milestones add the second encoder, fusion
and MSU. Anything the config asks for that is not implemented raises
``NotImplementedError`` with the milestone that will implement it, so an
experiment can never silently train a different architecture than it claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor, nn

from src.models.convnext_enc import ConvNeXtEncoder
from src.models.decoder import ChannelAlignment, PlainDecoder

#: Fusion modes implemented so far. Stages 1-2 concat and 3-4 cross-attention
#: arrive in Milestone 4, the Meta-Polyp branch in Milestone 3.
IMPLEMENTED_FUSION: frozenset[str] = frozenset({"single_a"})


def _stage_modes(fusion: str | Sequence[str]) -> list[str]:
    """Normalise the config's fusion setting to one mode per stage.

    Args:
        fusion: Either one mode for all stages (``"single_a"``, ``"all_concat"``,
            ...) or a list of 4 modes, stage 1 to stage 4.

    Returns:
        List of 4 mode strings.

    Raises:
        ValueError: if the value is neither a string nor a 4-entry sequence.
    """
    if isinstance(fusion, str):
        return [fusion] * 4
    modes = list(fusion)
    if len(modes) != 4:
        raise ValueError(f"fusion must be a string or 4 modes, got {modes}")
    return modes


class PolypSegNet(nn.Module):
    """Single-encoder baseline: ConvNeXt -> alignment -> plain decoder -> logits.

    Args:
        encoder_a: Branch A encoder exposing ``forward -> list[Tensor]`` and
            ``.out_channels``.
        channels: Fused width per stage ``C1..C4`` (stages 1-4). The alignments
            project ConvNeXt's widths onto these.
        encoder_b: Branch B encoder. ``None`` in Milestone 2.
        fusion: Per-stage fusion modes. Only all-``single_a`` is implemented.

    Forward:
        ``x: (B, 3, H, W)`` -> logits ``(B, 1, H, W)``.
    """

    def __init__(
        self,
        encoder_a: nn.Module,
        channels: Sequence[int] = (64, 128, 320, 512),
        encoder_b: nn.Module | None = None,
        fusion: str | Sequence[str] = "single_a",
    ) -> None:
        super().__init__()
        modes = _stage_modes(fusion)
        unsupported = sorted({mode for mode in modes if mode not in IMPLEMENTED_FUSION})
        if unsupported:
            raise NotImplementedError(
                f"fusion modes {unsupported} are not implemented yet "
                "(single_b arrives in Milestone 3, concat/xattn in Milestone 4)"
            )
        if encoder_b is not None:
            raise NotImplementedError("Branch B (Meta-Polyp encoder) arrives in Milestone 3")

        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.fusion = modes
        self.channels = [int(c) for c in channels]

        self.align_a = nn.ModuleList(
            [ChannelAlignment(int(c_in), c_out) for c_in, c_out in zip(encoder_a.out_channels, self.channels)]
        )
        self.decoder = PlainDecoder(self.channels)

    def forward(self, x: Tensor) -> Tensor:
        """Segment one batch.

        Args:
            x: ``(B, 3, H, W)`` normalised input, H and W divisible by 32.

        Returns:
            Logits ``(B, 1, H, W)``. Apply sigmoid outside the model.

        Raises:
            ValueError: if the encoder returns something other than 4 maps.
        """
        stage_maps = self.encoder_a(x)
        if len(stage_maps) != 4:
            raise ValueError(f"Encoder returned {len(stage_maps)} maps, expected 4")
        aligned = [align(feature) for align, feature in zip(self.align_a, stage_maps)]
        return self.decoder(aligned)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into the differential-learning-rate groups of section 4.

        Returns:
            Mapping with keys ``encoder_a``, ``encoder_b`` and ``head``. Groups may
            be empty (``encoder_b`` in Milestone 2); the caller drops empties.
            Every parameter of the model appears in exactly one group.
        """
        groups: dict[str, list[nn.Parameter]] = {
            "encoder_a": list(self.encoder_a.parameters()),
            "encoder_b": list(self.encoder_b.parameters()) if self.encoder_b is not None else [],
            "head": list(self.align_a.parameters()) + list(self.decoder.parameters()),
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
        cfg: Resolved config; reads ``model.encoder_a``, ``model.pretrained_a``,
            ``model.channels``, ``model.fusion``, ``model.msu``,
            ``model.msu_mode`` and ``model.skip_convformer``.

    Returns:
        The assembled model, on CPU, in train mode.

    Raises:
        NotImplementedError: if the config asks for a component that a later
            milestone owns (MSU, the Convformer skip block, a second encoder).
    """
    model_cfg = cfg["model"]

    if bool(model_cfg["skip_convformer"]):
        raise NotImplementedError(
            "model.skip_convformer=true is not implemented: the paper's Convformer skip block "
            "is deliberately out of scope (AGENTS.md section 11). Leave it false."
        )

    modes = _stage_modes(model_cfg["fusion"])
    if any(mode == "single_b" for mode in modes):
        raise NotImplementedError(
            "model.fusion: single_b needs the Meta-Polyp encoder, which arrives in Milestone 3"
        )
    if bool(model_cfg["msu"]) or any(mode != "single_a" for mode in modes):
        if bool(model_cfg["msu"]) and model_cfg["msu_mode"] not in {"paper", "released"}:
            raise ValueError(f"Unknown msu_mode {model_cfg['msu_mode']!r}")
        raise NotImplementedError(
            "This config needs fusion and/or MSU, which arrive in Milestone 4 (fusion) and "
            "Milestone 5 (MSU). Milestone 2 only builds the ConvNeXt baseline: "
            "model.fusion: single_a and model.msu: false. For the current milestone use "
            "configs/exp001_convnext_base_baseline.yaml."
        )

    encoder_a = ConvNeXtEncoder(
        model_name=str(model_cfg["encoder_a"]),
        pretrained=bool(model_cfg["pretrained_a"]),
    )
    return PolypSegNet(
        encoder_a=encoder_a,
        channels=model_cfg["channels"],
        encoder_b=None,
        fusion=modes,
    )
