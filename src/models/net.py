"""The assembled segmentation model and its parameter groups.

Milestone 3 builds one branch at a time: ConvNeXt (Branch A) or the Meta-Polyp
CAFormer encoder (Branch B), each followed by channel alignment and the same
plain decoder. Fusion of the two branches and MSU arrive in Milestones 4-5.
Anything the config asks for that is not implemented raises
``NotImplementedError`` with the milestone that will implement it, so an
experiment can never silently train a different architecture than it claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor, nn

from src.models.convnext_enc import ConvNeXtEncoder
from src.models.decoder import ChannelAlignment, PlainDecoder
from src.models.metapolyp_enc import MetaPolypEncoder

#: Fusion modes implemented so far. Stages 1-2 concat and 3-4 cross-attention
#: arrive in Milestone 4.
IMPLEMENTED_FUSION: frozenset[str] = frozenset({"single_a", "single_b"})
#: Encoder choices accepted by ``model.encoder``.
ENCODER_KINDS: frozenset[str] = frozenset({"convnext", "caformer"})


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
    """Single-branch baseline: encoder -> alignment -> plain decoder -> logits.

    Args:
        encoder_a: Branch A encoder exposing ``forward -> list[Tensor]`` and
            ``.out_channels``. Required for ``fusion=single_a``; pass ``None``
            for the Branch B baseline.
        channels: Fused width per stage ``C1..C4`` (stages 1-4). The alignment
            projects the active encoder's widths onto these.
        encoder_b: Branch B encoder. Required for ``fusion=single_b``; ``None``
            for the ConvNeXt baseline.
        fusion: Per-stage fusion modes. Only all-``single_a`` or all-``single_b``
            is implemented; dual-branch fusion is Milestone 4.

    Forward:
        ``x: (B, 3, H, W)`` -> logits ``(B, 1, H, W)``.

    Raises:
        ValueError: if the encoder required by ``fusion`` is missing.
        NotImplementedError: if both encoders are supplied (dual branch is
            Milestone 4) or a stage mixes modes.
    """

    def __init__(
        self,
        encoder_a: nn.Module | None,
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
                "(concat/xattn arrive in Milestone 4)"
            )
        if len(set(modes)) != 1:
            raise NotImplementedError(
                "one model for both branches needs per-stage fusion, which arrives in Milestone 4"
            )

        self.fusion = modes
        self.channels = [int(c) for c in channels]
        self.decoder = PlainDecoder(self.channels)

        if modes[0] == "single_a":
            if encoder_a is None:
                raise ValueError("fusion=single_a requires encoder_a")
            if encoder_b is not None:
                raise NotImplementedError(
                    "two encoders at once are not implemented; fusion arrives in Milestone 4"
                )
            self.encoder_a: nn.Module | None = encoder_a
            self.encoder_b: nn.Module | None = None
            self.align_a: nn.ModuleList | None = nn.ModuleList(
                [
                    ChannelAlignment(int(c_in), c_out)
                    for c_in, c_out in zip(encoder_a.out_channels, self.channels)
                ]
            )
            self.align_b: nn.ModuleList | None = None
        else:
            if encoder_b is None:
                raise ValueError("fusion=single_b requires encoder_b")
            if encoder_a is not None:
                raise NotImplementedError(
                    "two encoders at once are not implemented; fusion arrives in Milestone 4"
                )
            self.encoder_a = None
            self.encoder_b = encoder_b
            self.align_a = None
            self.align_b = nn.ModuleList(
                [
                    ChannelAlignment(int(c_in), c_out)
                    for c_in, c_out in zip(encoder_b.out_channels, self.channels)
                ]
            )

    def forward(self, x: Tensor) -> Tensor:
        """Segment one batch.

        Args:
            x: ``(B, 3, H, W)`` normalised input, H and W divisible by 32.

        Returns:
            Logits ``(B, 1, H, W)``. Apply sigmoid outside the model.

        Raises:
            ValueError: if the encoder returns something other than 4 maps.
            RuntimeError: if the module was assembled without an active branch.
        """
        encoder = self.encoder_a if self.encoder_a is not None else self.encoder_b
        alignments = self.align_a if self.align_a is not None else self.align_b
        if encoder is None or alignments is None:
            raise RuntimeError("No active encoder/alignment branch")
        stage_maps = encoder(x)
        if len(stage_maps) != 4:
            raise ValueError(f"Encoder returned {len(stage_maps)} maps, expected 4")
        aligned = [align(feature) for align, feature in zip(alignments, stage_maps)]
        return self.decoder(aligned)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Split parameters into the differential-learning-rate groups of section 4.

        Returns:
            Mapping with keys ``encoder_a``, ``encoder_b`` and ``head``. The
            inactive encoder group is empty; the caller drops empties. Every
            parameter of the model appears in exactly one group.
        """
        head: list[nn.Parameter] = list(self.decoder.parameters())
        if self.align_a is not None:
            head += list(self.align_a.parameters())
        if self.align_b is not None:
            head += list(self.align_b.parameters())
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
            ``model.fusion``, ``model.msu``, ``model.msu_mode`` and
            ``model.skip_convformer``.

    Returns:
        The assembled model, on CPU, in train mode.

    Raises:
        ValueError: if ``model.encoder`` is unknown or does not match ``fusion``.
        NotImplementedError: if the config asks for a component that a later
            milestone owns (dual-branch fusion, MSU, the Convformer skip block).
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

    modes = _stage_modes(model_cfg["fusion"])
    if bool(model_cfg["msu"]):
        if model_cfg["msu_mode"] not in {"paper", "released"}:
            raise ValueError(f"Unknown msu_mode {model_cfg['msu_mode']!r}")
        raise NotImplementedError(
            "MSU is not implemented yet (Milestone 5); leave model.msu false for the baselines"
        )
    unsupported = sorted({mode for mode in modes if mode not in IMPLEMENTED_FUSION})
    if unsupported:
        raise NotImplementedError(
            f"fusion modes {unsupported} are not implemented yet (Milestone 4); "
            "the baselines use fusion: single_a or single_b for every stage"
        )

    if encoder_kind == "convnext":
        if any(mode != "single_a" for mode in modes):
            raise NotImplementedError(
                "model.encoder=convnext requires fusion all single_a; "
                "dual-branch fusion arrives in Milestone 4"
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

    if any(mode != "single_b" for mode in modes):
        raise ValueError(
            "model.encoder=caformer requires fusion all single_b "
            f"(one branch only), got {modes}"
        )
    encoder_b = MetaPolypEncoder(
        model_name=str(model_cfg["encoder_b"]),
        pretrained=bool(model_cfg["pretrained_b"]),
    )
    return PolypSegNet(
        encoder_a=None,
        channels=model_cfg["channels"],
        encoder_b=encoder_b,
        fusion=modes,
    )
