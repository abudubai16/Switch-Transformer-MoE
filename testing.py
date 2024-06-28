import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from modules import MoELayer

from typing import Optional


class TransformerDecoderLayerMoE(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_projections: int,
        loss_scale: float = 3.5e3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.moe_layer = MoELayer(
            in_features=d_model, num_projections=num_projections, loss_scale=loss_scale
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x, loss = self.moe_layer(x)
        x = self.dropout(F.relu(x))
        return x, loss

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            logits, loss = self._ff_block(self.norm3(x))
            x = x + logits
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            logits, loss = self._ff_block(self.norm3(x))
            x = x + logits

        return x, loss


if __name__ == "__main__":
    a = torch.rand(64, 22, 256)
    layer = TransformerDecoderLayerMoE(256, 4, 4)
    logits, loss = layer(a, a)
    print(logits)
    print(loss)

    pass
