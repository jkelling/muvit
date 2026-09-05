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
#      <int k>                   -> exactly k levels on stage 0 (0 = old layout
#                                   with the whole decoder on stage 1; L =
#                                   whole decoder on stage 0, only final + loss
#                                   on stage 1)
#      'loss' / 'all' / 'last' / 'model' -> move the boundary all the way back:
#                                   the WHOLE model runs on stage 0 and only
#                                   the loss is computed on stage 1
# The remaining levels are decoded on stage 1 together with the final
# projection and the loss.
_LOSS_ONLY = "LOSS_ONLY"


def _boundary_levels(mae):
    n_levels = len(mae.encoder.levels)
    raw = _os.environ.get("MUVIT_PIPE_BOUNDARY", "").strip()
    if raw.lower() in ("loss", "all", "last", "model"):
        return _LOSS_ONLY
    if raw.lower() in ("", "auto", "balanced", "half"):
        return ((n_levels + 1) // 2 if
                (mae.decoder_mode in ("multi", "multi_iso")) else 0)
    try:
        k = int(raw)
    except ValueError:
        raise ValueError(
            f"MUVIT_PIPE_BOUNDARY must be an integer level count, 'auto' or "
            f"'loss', got {raw!r}.")
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


def _mask_select(mae, x, bbox):
    """Replicate the encoder mask-selection prefix (patch_embed + masking +
    retained-token selection) WITHOUT running any transformer layer, so the 6
    encoder blocks can be executed on different pipeline stages.

    Mirrors ``MuViTEncoder.forward_masked`` up to (but excluding) its
    ``self.layers`` loop. Returns ``(x, level_idx, coords_sel, coords_full,
    patches, batch_range, idx_retain, idx_mask)`` where ``x`` is the retained
    token embeddings to feed encoder layer 0 and ``coords_full`` is the
    unselected coords needed by the decode stages.
    """
    enc = mae.encoder
    x, patches, coords = enc.patch_embed(x, bbox)
    B, N, D = x.shape
    npl = N // len(enc.levels)
    level_idx = (torch.arange(
        patches.shape[1], device=x.device).unsqueeze(0).repeat(B, 1) // npl)

    masking_ratio = mae.masking_ratio
    masking_mode = mae.masking_mode
    N_retained = int(N * (1 - masking_ratio))
    if masking_mode == "dirichlet":
        prob_weights = torch.distributions.Dirichlet(
            torch.ones(len(enc.levels), device=x.device) * 0.5,
            validate_args=False).sample()
    elif masking_mode == "random":
        prob_weights = torch.ones(len(enc.levels), device=x.device)
    elif isinstance(masking_mode, (tuple, list)):
        prob_weights = torch.tensor(masking_mode, device=x.device)
    else:
        raise ValueError(f"Invalid masking mode: {masking_mode}")
    prob_per_block = torch.repeat_interleave(prob_weights, npl)
    idx = torch.stack([
        torch.multinomial(prob_per_block, N, replacement=False)
        for _ in range(B)
    ],
                      dim=0)
    idx_retain, idx_mask = idx[:, :N_retained], idx[:, N_retained:]
    batch_range = torch.arange(B, device=x.device)[:, None]
    x = x[batch_range, idx_retain]
    level_idx = level_idx[batch_range, idx_retain]
    return (x, level_idx, coords[batch_range, idx_retain], coords, patches,
            batch_range, idx_retain, idx_mask)


def _contig_state(state):
    out = []
    for t in state:
        if isinstance(t, torch.Tensor):
            if t.dtype.is_floating_point and not t.requires_grad:
                t = t.requires_grad_()
            t = t.contiguous()
        out.append(t)
    return tuple(out)


class _BlockStage(nn.Module):
    """Transformer-block-granular MuViT pipeline stage.

    One such module spans zero or more of: encoder transformer blocks, the
    assemble step, per-level decoder blocks, and (on the last stage) the final
    projection. ``spec`` is a dict with:

      first : bool        -- stage consumes the raw ``(x, bbox)`` input
      e0,e1 : int|None    -- encoder block range ``[e0, e1)`` (from the 6)
      assemble : bool     -- run the mask-token assembly after the encoder
      d0,d1  : int|None   -- decoder *level* range ``[d0, d1)``
      final : bool        -- last stage: project ``final(z)`` (loss_fn follows)

    Carry formats (all plain tuples of tensors):
      8-tuple mid-encoder: (x, level_idx, coords_sel, coords_full, patches,
                            batch_range, idx_retain, idx_mask)
      5-tuple assembled:   (z, coords_full, patches, batch_range, idx_mask)
      6-tuple mid-decode:  (zA, z, coords_full, patches, batch_range, idx_mask)
    """

    def __init__(self, mae, spec):
        super().__init__()
        self.mae = mae
        self.spec = spec

    def forward(self, state):
        s = self.spec
        if s.get("first"):
            x, bbox = state
            st = _mask_select(self.mae, x, bbox)
            e1 = s.get("e1", 0)
            for layer in self.mae.encoder.layers[:e1]:
                st = _run_enc_layer(self.mae, layer, st)
        else:
            st = list(state)
        # remaining encoder blocks
        if s.get("e0") is not None and not s.get("first"):
            x, level_idx, coords_sel, coords_full, patches, br, ir, im = st
            for layer in self.mae.encoder.layers[s["e0"]:s["e1"]]:
                x = layer(x,
                          level_idx=level_idx,
                          coords=coords_sel,
                          attention_mode=self.mae.encoder.attention_mode)
            st = [x, level_idx, coords_sel, coords_full, patches, br, ir, im]
        # assemble (end of encoder phase)
        if s.get("assemble"):
            x, _lvl, csel, coords_full, patches, br, ir, im = st
            Ni = patches.shape[1]
            npl = Ni // len(self.mae.encoder.levels)
            z = _assemble_z(self.mae, x, coords_full, Ni, npl, br, ir, im)
            st = [z, coords_full, patches, br, im]
        # decoder levels
        if s.get("d0") is not None:
            if len(st) == 5:
                z, coords_full, patches, br, im = st
                zA = None
            else:
                zA, z, coords_full, patches, br, im = st
            npl = z.shape[1] // len(self.mae.encoder.levels)
            chunk = _decode_levels(self.mae, z, coords_full, npl, s["d0"],
                                   s["d1"])
            zA = chunk if zA is None else torch.cat([zA, chunk], dim=1)
            if s.get("final"):
                return (self.mae.final(zA).contiguous(), patches.contiguous())
            st = [zA, z, coords_full, patches, br, im]
        if s.get("final"):
            z, coords_full, patches, br, im = st
            return (self.mae.final(z).contiguous(), patches.contiguous())
        return _contig_state(st)


def _run_enc_layer(mae, layer, st):
    x, level_idx, coords_sel, coords_full, patches, br, ir, im = st
    x = layer(x,
              level_idx=level_idx,
              coords=coords_sel,
              attention_mode=mae.encoder.attention_mode)
    return (x, level_idx, coords_sel, coords_full, patches, br, ir, im)


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
        # DeepSpeed's inter-stage activation P2P requires dense, non-overlapping
        # tensors (the encoder output contains advanced-indexing views), so make
        # everything contiguous before handing it across the pipeline. `patches`
        # are carried so the last stage can compute MSE(input, output) with a
        # target that is consistent across ranks by construction. Token counts
        # are NOT transported: downstream stages derive N/npl from tensor
        # ``.shape`` (metadata, no host sync), avoiding ``Tensor.item()``.
        return (y.contiguous(), coords.contiguous(), patches.contiguous(),
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
        mask_tokens = torch.repeat_interleave(mae.mask_token,
                                              N_per_level,
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
        y, coords, patches, batch_range, idx_retain, idx_mask = enc_state
        Ni = patches.shape[1]
        N_per_level = Ni // len(self.mae.encoder.levels)
        z = _assemble_z(self.mae, y, coords, Ni, N_per_level, batch_range,
                        idx_retain, idx_mask)
        return z, coords, patches, batch_range, idx_mask


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
            out.append(mae.decoder[i](zs[i],
                                      cs[i],
                                      context=z,
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
        z, coords, patches, _batch_range, _idx_mask = state
        npl = z.shape[1] // len(self.mae.encoder.levels)
        if self.mae.decoder_mode == "single":
            z = self.mae.decoder(z, coords)
        elif self.mae.decoder_mode == "multi":
            zs = torch.split(z, npl, dim=1)
            cs = torch.split(coords, npl, dim=1)
            z = torch.cat(
                [
                    self.mae.decoder[i](
                        _z, _c, context=z, context_coords=coords)
                    for i, (_z, _c) in enumerate(zip(zs, cs))
                ],
                dim=1,
            )
        elif self.mae.decoder_mode == "multi_iso":
            zs = torch.split(z, npl, dim=1)
            cs = torch.split(coords, npl, dim=1)
            z = torch.cat(
                [
                    self.mae.decoder[i](_z, _c)
                    for i, (_z, _c) in enumerate(zip(zs, cs))
                ],
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
                patches.contiguous(), batch_range.contiguous(),
                idx_mask.contiguous())


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
        zA, z, coords, patches, _batch_range, _idx_mask = state
        npl = z.shape[1] // len(self.mae.encoder.levels)
        if self.first >= len(self.mae.encoder.levels):
            # Boundary moved all level-decoders onto stage 0: nothing left to
            # decode here, just project the tokens received from stage 0.
            z = zA
        else:
            zB = _decode_levels(self.mae, z, coords, npl, self.first,
                                len(self.mae.encoder.levels))
            z = torch.cat([zA, zB], dim=1)
        return self.mae.final(z), patches


class LossOnlyStage(nn.Module):
    """Pipeline final stage: only the reconstruction loss is computed here.

    Receives ``(z, patches)`` (the decoder output already projected to patch
    space by the input stage) and passes them through unchanged; DeepSpeed
    applies :func:`mae_pipeline_loss` (MSE) to the last stage output. Used for
    the 'boundary fully back' layout where the whole model runs on stage 0 and
    stage 1 only performs the loss computation.
    """

    def __init__(self):
        super().__init__()
        # DeepSpeed's engine init expects every stage to own at least one
        # trainable parameter (optimizer/param-group setup syncs across
        # stages); keep a scalar placeholder so a loss-only final stage does
        # not deadlock the pipeline engine.
        self._placeholder = nn.Parameter(torch.zeros(()))

    def forward(self, state):
        return state


def _encode_assemble_decode_all(mae, x_bbox):
    """Encode, assemble and decode ALL level-decoders on a single stage.

    Returns ``(z_all, patches)`` with ``z_all`` the concatenated decoded
    tokens of every level (both gradients flow, so this is a valid DeepSpeed
    cross-stage hand-off) and ``patches`` the input-derived target.
    """
    x, bbox = x_bbox
    y, coords, patches, batch_range, idx_retain, idx_mask = _forward_masked(
        mae, x, bbox)
    for t in (y, coords, patches):
        if t.dtype.is_floating_point and not t.requires_grad:
            t = t.requires_grad_()
    Ni = int(patches.shape[1])
    npl = Ni // len(mae.encoder.levels)
    z = _assemble_z(mae, y, coords, Ni, npl, batch_range, idx_retain, idx_mask)
    z = _decode_levels(mae, z, coords, npl, 0, len(mae.encoder.levels))
    return z.contiguous(), patches.contiguous()


class EncoderAllDecodeStage(nn.Module):
    """Boundary back: encode + assemble + decode ALL levels on stage 0.

    Stage 1 (:class:`FinalLossStage`) projects the received tokens to patch
    space and computes the loss. Only ``(z_all, patches)`` cross the pipeline,
    so every cross-stage activation receives a gradient (DeepSpeed invariant).
    """

    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def drain(self):
        return

    def forward(self, x_bbox):
        return _encode_assemble_decode_all(self.mae, x_bbox)


class EncoderAllDecodeFinalStage(nn.Module):
    """Balanced 2-stage stage 0: entire model except the loss.

    Encodes, assembles, decodes ALL level-decoders and runs the final patch
    projection on the input stage; only the scalar MSE loss stays on stage 1
    (:class:`LossOnlyStage` + :func:`mae_pipeline_loss`). Outputs
    ``(z, patches)`` with ``z`` the (differentiable) projected decoder tokens.
    """

    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def drain(self):
        return

    def forward(self, x_bbox):
        z, patches = _encode_assemble_decode_all(self.mae, x_bbox)
        return self.mae.final(z).contiguous(), patches


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


def _block_specs(mae, num_stages):
    """Contiguous transformer-block partition specs for ``_BlockStage``.

    Units: P (patch_embed+mask select), E_i (encoder blocks), A (assemble),
    D_i (per-level decoder groups), F (final). The list is split into
    ``num_stages`` contiguous, near-equal slices — every boundary falls between
    whole transformer blocks. The assembly step A lands inside the slice that
    also carries the last encoder block E_{n-1} for these sizes, so no stage is
    left with only trivial (~zero load) work; F is on the last slice with its
    decoder group + the (cheap) loss.
    """
    n_lvl = len(mae.encoder.levels)
    units = (["P"] + [f"E{i}" for i in range(len(mae.encoder.layers))] +
             ["A"] + [f"D{i}" for i in range(n_lvl)] + ["F"])
    L = len(units)
    base, rem = divmod(L, num_stages)
    sizes = [base + 1] * rem + [base] * (num_stages - rem)
    specs = []
    cut = 0
    for sz in sizes:
        seg = units[cut:cut + sz]
        cut += sz
        spec = {}
        e = [int(u[1:]) for u in seg if u.startswith("E")]
        d = [int(u[1:]) for u in seg if u.startswith("D")]
        if "P" in seg:
            spec["first"] = True
            spec["e1"] = len(e)  # first stage runs encoder layers [0, len(e))
        elif e:
            spec["e0"] = min(e)
            spec["e1"] = max(e) + 1
        if "A" in seg:
            spec["assemble"] = True
        if d:
            spec["d0"] = min(d)
            spec["d1"] = max(d) + 1
        if "F" in seg:
            spec["final"] = True
        specs.append(spec)
    return specs


def build_mae3d_pipeline_blocks(mae, num_stages: int):
    """Transformer-block-granular MuViT pipeline stages (2/3/4).

    Splits the 6 encoder blocks and the per-level decoder blocks across the
    stages at whole-transformer-block boundaries (see ``_block_specs``), so
    every stage -- including the last, which does decoder + `final` + loss --
    carries real work.
    """
    if num_stages not in (2, 3, 4):
        raise ValueError(
            f"MuViT block pipeline supports num_stages in (2, 3, 4), got "
            f"{num_stages}.")
    return [_BlockStage(mae, sp) for sp in _block_specs(mae, num_stages)]


def build_mae3d_pipeline(mae, num_stages: int):
    """Return ``num_stages`` callable pipeline stages for a ``MuViTMAE3d``.

    The last stage outputs the (differentiable) decoder tokens ``z``; the
    scalar reconstruction MSE is computed against the DeepSpeed ``labels``
    (input-derived target patches) via :func:`mae_pipeline_loss`.

    Raises ``ValueError`` for unsupported stage counts.
    """
    if _os.environ.get("MUVIT_PIPE_SPLIT") == "blocks":
        return build_mae3d_pipeline_blocks(mae, num_stages)

    if num_stages == 2:
        # The 2-stage boundary can be moved via MUVIT_PIPE_BOUNDARY (see
        # _boundary_levels): from 0 (encode only + whole decoder/final/loss on
        # stage 1) through balanced (half the level-decoders on each stage) to
        # L (whole decoder on stage 0, only final+loss on stage 1) and
        # 'loss' (the whole model on stage 0, only the loss on stage 1).
        n_levels = len(mae.encoder.levels)
        is_multi = mae.decoder_mode in ("multi", "multi_iso")
        first = _boundary_levels(mae)
        if first == _LOSS_ONLY:
            return [EncoderAllDecodeFinalStage(mae), LossOnlyStage()]
        if (is_multi and n_levels >= 2 and 0 < first < n_levels):
            return [
                EncoderHalfDecodeStage(mae, first),
                HalfDecodeFinalStage(mae, first)
            ]
        if (is_multi and n_levels >= 2 and first == n_levels):
            # Whole decoder on stage 0; only the final patch projection + loss
            # stay on stage 1.
            return [EncoderAllDecodeStage(mae), FinalLossStage(mae)]
        return [EncoderStage(mae), DecodeFinalStage(mae)]
    if num_stages == 3:
        return [
            EncoderStage(mae),
            AssembleDecodeStage(mae),
            FinalLossStage(mae)
        ]
    if num_stages == 4:
        return [
            EncoderStage(mae),
            AssembleStage(mae),
            DecodeStage(mae),
            FinalLossStage(mae)
        ]
    raise ValueError(
        f"MuViT MAE pipeline currently supports num_stages in (2, 3, 4), got {num_stages}."
    )
