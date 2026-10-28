"""Model building blocks.

Filled in milestone order (AGENTS.md section 9):

* ``convnext_enc.py`` / ``metapolyp_enc.py`` - encoders returning 4 maps at
  strides 4, 8, 16, 32 (milestone 2 and 3).
* ``fusion.py`` - concat and cross-attention fusion (milestone 4).
* ``msu.py`` / ``decoder.py`` - multi-scale upsampling and the U-Net decoder
  (milestone 5).
* ``net.py`` - the assembled model (milestone 2 onwards).

Nothing is implemented yet: Milestone 1 is data, metrics and losses only.
"""
