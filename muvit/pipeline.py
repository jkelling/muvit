"""Pipeline-parallel decomposition of the MuViT MAE model.

The monolithic forward pass of :class:`muvit.mae.MuViTMAE3d` is split into a
sequence of callable stages that can be handed to a pipeline-parallel wrapper
(e.g. DeepSpeed ``PipelineModule``):

- stage 0 (encoder + masking): ``(x, bbox) -> encoded state``
- optional further stages assemble the decoder input (mask tokens / retained
  tokens) and run the decoder
- the final stage projects the decoder output back to patch space

The last stage outputs the (differentiable) decoder tokens ``z``. The scalar
reconstruction MSE is computed against the input-derived target patches --
delivered to the last stage as DeepSpeed ``labels`` -- by
:func:`mae_pipeline_loss` (i.e. the loss is computed on the last stage as the
MSE between the model input and its reconstruction).

Cross-stage state only holds tensors that are either needed by the decoder
(``y``, ``coords``) or are plain integer book-keeping (``N``,
``batch_range``, ``idx_retain``, ``idx_mask``). DeepSpeed's activation
gradient-pass requires every *floating point* tensor handed to a later stage
to contribute a gradient, which is why the (gradient-neutral) ``patches``
target is *not* part of the pipeline state -- it travels as labels instead.

Supported stage counts:

- ``2``: ``[EncoderStage, DecodeFinalStage]``
- ``3``: ``[EncoderStage, AssembleDecodeStage, FinalLossStage]``
- ``4``: ``[EncoderStage, AssembleStage, DecodeStage, FinalLossStage]``
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "build_mae3d_pipeline",
    "mae_pipeline_loss",
    "EncoderStage",
    "AssembleStage",
    "DecodeStage",
    "FinalLossStage",
    "AssembleDecodeStage",
    "DecodeFinalStage",
]


import os as _os

# Stage-boundary option for the 2-stage layout: how many level-decoders are
# pulled onto the *input* stage (stage 0) of a 2-stage pipeline.
#   MUVIT_PIPE_BOUNDARY:
#      '' / 'auto' / 'balanced' -> ceil(L/2) if decoder_mode is multi/multi_iso
#                                 and there are >= 2 levels, else 0 (old layout)
#      <int k>                    -> exactly k levels on stage 0 (0 = old layout
#                                    with the whole decoder on stage 1; L = whole
#                                    decoder on stage 0)
# The remaining levels are decoded on stage 1 together with the final
# projection and the loss.
def _boundary_levels(mae):
    n_levels = len(mae.encoder.levels)
    raw = _os.environ.get("MUVIT_PIPE_BOUNDARY", "").strip()
    if raw.lower() in ("", "auto", "balanced", "half"):
        return ((n_levels + 1) // 2
                if (mae.decoder_mode in ("multi", "multi_iso")) else 0)
    try:
        k = int(raw)
    except ValueError:
        raise ValueError(
            f"MUVIT_PIPE_BOUNDARY must be an integer level count or "
            f"'auto', got {raw!r}.")
    return max(0, min(k, n_levels))


def _forward_masked(mae, x, bbox):
    """Run the encoder's masked forward, mirroring MuViTMAE3d.forward."""
    return mae.encoder.forward_masked(
        x,
        bbox,
        mae.masking_ratio,
        mae.masking_mode,
        False,
        None,
        masking_mode_is_ratio=False,
    )


class EncoderStage(nn.Module):
    """Pipeline stage 0: encode the (multi-level) input and sample the mask.

    Input:  ``(x, bbox)`` (the pyramid built by ``MuViTMaeRepo.build_pyramid``
    and its bounding box).
    Output: ``(y, coords, N, batch_range, idx_retain, idx_mask)`` with ``N``
    being a scalar ``int64`` tensor holding the total token count.
    """

    def __init__(self, mae):
        super().__init__()
        self.mae = mae
        self.target_dest: Optional[int] = None
        self.target_group: Optional[dist.ProcessGroup] = None
        self._pending: list = []

    def forward(self, x_bbox):
        x, bbox = x_bbox
        y, coords, patches, batch_range, idx_retain, idx_mask = _forward_masked(
            self.mae, x, bbox)
        if self.target_dest is not None:
            self._pending.append(
                dist.isend(patches.contiguous(),
                           dst=self.target_dest,
                           group=self.target_group))
        # DeepSpeed's inter-stage gradient pass requires every *floating point*
        # output of this stage to require grad (e.g. the grid coordinates are
        # usually grad-free by construction).
        for t in (y, coords, patches):
            if t.dtype.is_floating_point and not t.requires_grad:
                t = t.requires_grad_()
        N = torch.tensor(patches.shape[1], device=y.device, dtype=torch.long)
        # DeepSpeed's inter-stage activation P2P requires dense, non-overlapping
        # tensors (the encoder output contains advanced-indexing views), so make
        # everything contiguous before handing it across the pipeline. `patches`
        # are carried so the last stage can compute MSE(input, output) with a
        # target that is consistent across ranks by construction.
        return (y.contiguous(), coords.contiguous(), patches.contiguous(), N,
                batch_range.contiguous(), idx_retain.contiguous(),
                idx_mask.contiguous())

    def drain(self):
        """Wait for all outstanding target sends to complete."""
        pending, self._pending = self._pending, []
        for handle in pending:
            handle.wait()


def _assemble_z(mae, y, coords, N, N_per_level, batch_range, idx_retain,
                idx_mask):
    """Assemble decoder input tokens (mask tokens + retained encoder output)."""
    z = torch.zeros(y.shape[0], N, y.shape[-1], device=y.device, dtype=y.dtype)

    if mae.decoder_mode == "single":
        z[batch_range, idx_mask] = mae.mask_token
        z[batch_range, idx_retain] = y
    else:  # "multi" / "multi_iso"
        mask_tokens = torch.repeat_interleave(mae.mask_token, N_per_level,
                                              dim=1).repeat(y.shape[0], 1, 1)
        z[batch_range, idx_mask] = mask_tokens[batch_range, idx_mask]
        z[batch_range, idx_retain] = y

    return z


class AssembleStage(nn.Module):
    """Pipeline stage: assemble the decoder input from the encoded state."""

    def __init__(self, mae):
        super().__init__()
        self.mae = mae
        self.device = None

    def forward(self, enc_state):
        y, coords, patches, N, batch_range, idx_retain, idx_mask = enc_state
        Ni = int(N.item())
        N_per_level = Ni // len(self.mae.encoder.levels)
        z = _assemble_z(self.mae, y, coords, Ni, N_per_level, batch_range,
                        idx_retain, idx_mask)
        return (z, coords, patches,
                torch.tensor(N_per_level, device=y.device, dtype=torch.long),
                batch_range, idx_mask)



def _decode_levels(mae, z, coords, npl, start, stop):
    """Run MAE decoders for levels [start, stop) and concatenate their tokens.

    Under ``multi``/``multi_iso`` decoding each level attends over the *full*
    assembled token tensor ``z``/``coords`` (cross-attention to all levels), so
    individual levels can be decoded on different pipeline stages and the
    per-level outputs concatenated afterwards. ``npl`` = tokens per level.
    """
    zs = torch.split(z, npl, dim=1)
    cs = torch.split(coords, npl, dim=1)
    out = []
    mode = mae.decoder_mode
    for i in range(start, stop):
        if mode == "multi":
            out.append(mae.decoder[i](zs[i], cs[i], context=z,
                                      context_coords=coords))
        elif mode == "multi_iso":
            out.append(mae.decoder[i](zs[i], cs[i]))
        else:
            raise ValueError(
                f"Level-split decoding requires decoder_mode multi or "
                f"multi_iso, got {mode!r}.")
    return torch.cat(out, dim=1)


class DecodeStage(nn.Module):
    """Pipeline stage: run the MAE decoder(s); returns ``(decoded, patches)``."""

    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def forward(self, state):
        z, coords, patches, N_per_level, _batch_range, _idx_mask = state
        npl = int(N_per_level.item())
        if self.mae.decoder_mode == "single":
            z = self.mae.decoder(z, coords)
        elif self.mae.decoder_mode == "multi":
            zs = torch.split(z, npl, dim=1)
            cs = torch.split(coords, npl, dim=1)
            z = torch.cat(
                [
                    self.mae.decoder[i](_z, _c, context=z, context_coords=coords)
                    for i, (_z, _c) in enumerate(zip(zs, cs))
                ],
                dim=1,
            )
        elif self.mae.decoder_mode == "multi_iso":
            zs = torch.split(z, npl, dim=1)
            cs = torch.split(coords, npl, dim=1)
            z = torch.cat(
                [self.mae.decoder[i](_z, _c)
                 for i, (_z, _c) in enumerate(zip(zs, cs))],
                dim=1,
            )
        else:
            raise ValueError(f"Invalid decoder mode: {self.mae.decoder_mode}")
        return z, patches



class EncoderHalfDecodeStage(nn.Module):
    """Balanced 2-stage stage 0: encode + assemble + decoder levels [0, first).

    Keeps half of the (parallel) level-decoders on the input stage so the
    compute load is shared between the two GBU stages instead of pushing the
    whole decoder onto stage 1 (GPU1 100% vs GPU0 ~0%).
    Outputs: ``(zA, z, coords, patches, N_per_level, batch_range, idx_mask)``
    where ``zA`` are the decoded tokens of the first ``first`` levels and ``z``
    is the full assembled token tensor still needed by the remaining levels.
    """

    def __init__(self, mae, first):
        super().__init__()
        self.mae = mae
        self.first = first

    def drain(self):
        # No explicit async target sends on this stage; hook kept for the
        # driver's uniform input-stage drain() call.
        return

    def forward(self, x_bbox):
        x, bbox = x_bbox
        y, coords, patches, batch_range, idx_retain, idx_mask = _forward_masked(
            self.mae, x, bbox)
        for t in (y, coords, patches):
            if t.dtype.is_floating_point and not t.requires_grad:
                t = t.requires_grad_()
        Ni = int(patches.shape[1])
        npl = Ni // len(self.mae.encoder.levels)
        z = _assemble_z(self.mae, y, coords, Ni, npl, batch_range, idx_retain,
                        idx_mask)
        zA = _decode_levels(self.mae, z, coords, npl, 0, self.first)
        return (zA.contiguous(), z.contiguous(), coords.contiguous(),
                patches.contiguous(),
                torch.tensor(npl, device=y.device, dtype=torch.long),
                batch_range.contiguous(), idx_mask.contiguous())


class HalfDecodeFinalStage(nn.Module):
    """Balanced 2-stage stage 1: decoder levels [first, L) + final + loss.

    Decodes the remaining levels (cross-attending over the full ``z``/``coords``
    received from stage 0), concatenates with ``zA`` and returns the projected
    tokens and the target patches for :func:`mae_pipeline_loss`.
    """

    def __init__(self, mae, first):
        super().__init__()
        self.mae = mae
        self.first = first

    def forward(self, state):
        zA, z, coords, patches, N_per_level, _batch_range, _idx_mask = state
        npl = int(N_per_level.item())
        zB = _decode_levels(self.mae, z, coords, npl, self.first,
                            len(self.mae.encoder.levels))
        z = torch.cat([zA, zB], dim=1)
        return self.mae.final(z), patches


class FinalLossStage(nn.Module):
    """Final pipeline stage: project decoder tokens to patch space.

    Returns ``(z, patches)`` -- the (differentiable) decoder output and the
    input-derived target patches carried through the pipeline state -- so the
    raw reconstruction MSE between input and output is computed on the last
    stage by :func:`mae_pipeline_loss` (DeepSpeed ``loss_fn``).
    """

    def __init__(self, mae, eps: float = 1e-2):
        super().__init__()
        self.mae = mae
        self.eps = eps
        self.target_source: Optional[int] = None
        self.target_group: Optional[dist.ProcessGroup] = None

    def forward(self, state):
        z, patches = state
        return self.mae.final(z), patches


def mae_pipeline_loss(output, labels):
    """DeepSpeed pipeline ``loss_fn``: MSE between the decoder output and the
    input-derived target patches carried through the pipeline state. ``labels``
    are ignored (DeepSpeed still delivers them from each stage's data iterator
    on every rank) because the target already travels inside ``output``, which
    keeps input and output consistent across ranks by construction."""
    z, patches = output
    return F.mse_loss(z, patches)


class AssembleDecodeStage(nn.Module):
    """Merge of :class:`AssembleStage` + :class:`DecodeStage` (3-stage config)."""

    def __init__(self, mae):
        super().__init__()
        self.assemble = AssembleStage(mae)
        self.decode = DecodeStage(mae)

    def forward(self, enc_state):
        return self.decode(self.assemble(enc_state))


class DecodeFinalStage(nn.Module):
    """Merge of assemble + decode + final projection (2-stage config).

    Returns the (differentiable) decoder output tensor ``z``; the scalar MSE
    loss is computed by ``mae_pipeline_loss`` on the last stage.
    """

    def __init__(self, mae, eps: float = 1e-2):
        super().__init__()
        self.assemble = AssembleStage(mae)
        self.decode = DecodeStage(mae)
        self.final = FinalLossStage(mae, eps=eps)

    def forward(self, enc_state):
        return self.final(self.decode(self.assemble(enc_state)))


def build_mae3d_pipeline(mae, num_stages: int):
    """Return ``num_stages`` callable pipeline stages for a ``MuViTMAE3d``.

    The last stage outputs the (differentiable) decoder tokens ``z``; the
    scalar reconstruction MSE is computed against the DeepSpeed ``labels``
    (input-derived target patches) via :func:`mae_pipeline_loss`.

    Raises ``ValueError`` for unsupported stage counts.
    """
    if num_stages == 2:
        # The 2-stage boundary can be moved via MUVIT_PIPE_BOUNDARY (see
        # _boundary_levels): by default the level-decoders are split evenly so
        # encode+assemble+half the decoding run on stage 0 and the remaining
        # decoding + final + loss on stage 1. Boundary 0/L fall back to the
        # conventional encoder-only / decoder-only layout.
        first = _boundary_levels(mae)
        n_levels = len(mae.encoder.levels)
        if (mae.decoder_mode in ("multi", "multi_iso") and n_levels >= 2
                and 0 < first < n_levels):
            return [EncoderHalfDecodeStage(mae, first),
                    HalfDecodeFinalStage(mae, first)]
        return [EncoderStage(mae), DecodeFinalStage(mae)]
    if num_stages == 3:
        return [EncoderStage(mae), AssembleDecodeStage(mae), FinalLossStage(mae)]
    if num_stages == 4:
        return [EncoderStage(mae), AssembleStage(mae), DecodeStage(mae),
                FinalLossStage(mae)]
    raise ValueError(
        f"MuViT MAE pipeline currently supports num_stages in (2, 3, 4), got {num_stages}.")
