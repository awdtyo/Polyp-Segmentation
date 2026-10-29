"""Model building blocks.

Filled in milestone order (AGENTS.md section 9):

* ``convnext_enc.py`` / ``metapolyp_enc.py`` - encoders returning 4 maps at
  strides 4, 8, 16, 32 (milestones 2 and 3).
* ``fusion.py`` - concat and cross-attention fusion (milestone 4).
* ``msu.py`` / ``decoder.py`` - multi-scale upsampling and the U-Net decoder
  (milestone 5).
* ``net.py`` - the assembled model (milestone 2 onwards).

Milestones 1-5 are implemented: both single-branch baselines, channel alignment,
per-stage concat/cross-attention fusion, the U-Net decoder with optional
paper-mode MSU, and the assembled dual-branch model. The released MSU variant
and the Convformer skip block remain reserved and raise.
"""
