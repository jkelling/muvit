"""Pipeline-parallel decomposition of the MuViT MAE model.

The monolithic forward pass of :class:`muvit.mae.MuViTMAE3d` is split into a
sequence of callable stages that can be handed to a pipeline-parallel wrapper
(e.g. DeepSpeed ``PipelineModule``):

- stage 0 (encoder + masking): ``(x, bbox) -> encoded state``
- optional further stages assemble the decoder input (mask tokens / retained
  tokens) and run the decoder
- the final stage produces the reconstruction loss

The stages reference the parameters *in place* (they are thin wrappers around
the model's own submodules), so no parameters are duplicated and training
updates flow back into the original model.

Supported stage counts:

- ``2``: ``[EncoderStage, DecoderLossStage]``
- ``3``: ``[EncoderStage, AssembleDecodeStage, FinalLossStage]``
- ``4``: ``[EncoderStage, AssembleStage, DecodeStage, FinalLossStage]``
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "build_mae3d_pipeline",
    "EncoderStage",
    "AssembleStage",
    "DecodeStage",
    "FinalLossStage",
    "AssembleDecodeStage",
    "DecoderLossStage",
]


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
    and its bounding box, or ``None`` bbox).
    Output: ``(y, coords, patches, batch_range, idx_retain, idx_mask)``.
    """

    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def forward(self, x_bbox):
        x, bbox = x_bbox
        return _forward_masked(self.mae, x, bbox)


def _assemble_z(self, y, coords, patches, batch_range, idx_retain, idx_mask):
    """Assemble decoder input tokens (mask tokens + retained encoder output)."""
    N = patches.shape[1]
    N_per_level = N // len(self.encoder.levels)
    z = torch.zeros(patches.shape[0], N, y.shape[-1], device=patches.device,
                    dtype=patches.dtype)

    if self.decoder_mode == "single":
        z[batch_range, idx_mask] = self.mask_token
        z[batch_range, idx_retain] = y
    else:  # "multi" / "multi_iso"
        mask_tokens = torch.repeat_interleave(self.mask_token, N_per_level,
                                              dim=1).repeat(patches.shape[0], 1,
                                                            1)
        z[batch_range, idx_mask] = mask_tokens[batch_range, idx_mask]
        z[batch_range, idx_retain] = y

    return z, coords, patches, batch_range, idx_mask, N_per_level


class AssembleStage(nn.Module):
    """Pipeline stage: assemble the decoder input from the encoded state."""

    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def forward(self, enc_state):
        y, coords, patches, batch_range, idx_retain, idx_mask = enc_state
        return _assemble_z(self.mae, y, coords, patches, batch_range, idx_retain,
                           idx_mask)


class DecodeStage(nn.Module):
    """Pipeline stage: run the MAE decoder(s)."""

    def __init__(self, mae):
        super().__init__()
        self.mae = mae

    def forward(self, state):
        z, coords, patches, batch_range, idx_mask, N_per_level = state
        if self.mae.decoder_mode == "single":
            z = self.mae.decoder(z, coords)
        elif self.mae.decoder_mode == "multi":
            zs = torch.split(z, N_per_level, dim=1)
            cs = torch.split(coords, N_per_level, dim=1)
            z = torch.cat(
                [
                    self.mae.decoder[i](_z, _c, context=z, context_coords=coords)
                    for i, (_z, _c) in enumerate(zip(zs, cs))
                ],
                dim=1,
            )
        elif self.mae.decoder_mode == "multi_iso":
            zs = torch.split(z, N_per_level, dim=1)
            cs = torch.split(coords, N_per_level, dim=1)
            z = torch.cat(
                [self.mae.decoder[i](_z, _c) for i, (_z, _c) in enumerate(zip(zs, cs))],
                dim=1,
            )
        else:
            raise ValueError(f"Invalid decoder mode: {self.mae.decoder_mode}")
        return z, patches, batch_range, idx_mask


class FinalLossStage(nn.Module):
    """Pipeline stage: project to patch space, normalise and compute the loss."""

    def __init__(self, mae, eps: float = 1e-2):
        super().__init__()
        self.mae = mae
        self.eps = eps

    def forward(self, state):
        z, patches, batch_range, idx_mask = state
        z = self.mae.final(z)

        if self.mae.loss_fn == "norm_mse":
            p_mean = patches.mean(dim=-1, keepdim=True)
            p_std = patches.std(dim=-1, keepdim=True)
        elif self.mae.loss_fn in ("mse", "mse_fft"):
            p_mean = torch.tensor(0, device=patches.device, dtype=patches.dtype)
            p_std = torch.tensor(1 - self.eps, device=patches.device,
                                 dtype=patches.dtype)
        else:
            raise ValueError(f"Invalid loss: {self.mae.loss_fn}")

        patches_normed = (patches - p_mean) / (p_std + self.eps)

        loss = F.mse_loss(z[batch_range, idx_mask], patches_normed[batch_range, idx_mask])
        if self.mae.loss_fn == "mse_fft":
            z2 = self.mae.token_to_patch(z[batch_range, idx_mask]).to(torch.float32)
            patches2 = self.mae.token_to_patch(patches_normed[batch_range, idx_mask]).to(
                torch.float32)
            ndim = self.mae.ndim
            z2f = torch.fft.rfftn(z2, dim=tuple(-(i + 1) for i in range(ndim)))
            patches2f = torch.fft.rfftn(patches2, dim=tuple(-(i + 1) for i in range(ndim)))
            loss = loss + 0.01 * F.l1_loss(z2f, patches2f)

        return loss


class AssembleDecodeStage(nn.Module):
    """Merge of :class:`AssembleStage` + :class:`DecodeStage` (3-stage config)."""

    def __init__(self, mae):
        super().__init__()
        self.assemble = AssembleStage(mae)
        self.decode = DecodeStage(mae)

    def forward(self, enc_state):
        return self.decode(self.assemble(enc_state))


class DecoderLossStage(nn.Module):
    """Merge of assemble + decode + final-loss (2-stage config)."""

    def __init__(self, mae, eps: float = 1e-2):
        super().__init__()
        self.assemble = AssembleStage(mae)
        self.decode = DecodeStage(mae)
        self.final_loss = FinalLossStage(mae, eps=eps)

    def forward(self, enc_state):
        return self.final_loss(self.decode(self.assemble(enc_state)))


def build_mae3d_pipeline(mae, num_stages: int):
    """Return ``num_stages`` callable pipeline stages for a ``MuViTMAE3d``.

    Raises ``ValueError`` for unsupported stage counts.
    """
    if num_stages == 2:
        return [EncoderStage(mae), DecoderLossStage(mae)]
    if num_stages == 3:
        return [EncoderStage(mae), AssembleDecodeStage(mae), FinalLossStage(mae)]
    if num_stages == 4:
        return [EncoderStage(mae), AssembleStage(mae), DecodeStage(mae), FinalLossStage(mae)]
    raise ValueError(
        f"MuViT MAE pipeline currently supports num_stages in (2, 3, 4), got {num_stages}.")
