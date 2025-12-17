import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from typing import Optional, Tuple, Union


def patch_dropout_forward(
        x: ms.Tensor,
        prob: float,
        num_prefix_tokens: int,
        ordered: bool,
        training: bool,
) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
    """
    Common forward logic for patch dropout.

    Args:
        x: Input tensor of shape (B, L, D)
        prob: Dropout probability
        num_prefix_tokens: Number of prefix tokens to preserve
        ordered: Whether to maintain patch order
        training: Whether in training mode

    Returns:
        Tuple of (output tensor, keep_indices or None)
    """
    if not training or prob == 0.:
        return x, None

    if num_prefix_tokens:
        prefix_tokens, x = x[:, :num_prefix_tokens], x[:, num_prefix_tokens:]
    else:
        prefix_tokens = None

    B = x.shape[0]
    L = x.shape[1]
    num_keep = max(1, int(L * (1. - prob)))
    keep_indices = mint.argsort(mint.randn(B, L, device=x.device), dim=-1)[:, :num_keep]

    if ordered:
        # NOTE does not need to maintain patch order in typical transformer use,
        # but possibly useful for debug / visualization
        keep_indices = keep_indices.sort(dim=-1)[0]

    x = x.gather(1, keep_indices.unsqueeze(-1).expand((-1, -1) + x.shape[2:]))

    if prefix_tokens is not None:
        x = mint.cat((prefix_tokens, x), dim=1)

    return x, keep_indices


class PatchDropout(msnn.Cell):
    """
    Patch Dropout without returning indices.
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        output, _ = patch_dropout_forward(
            x,
            self.prob,
            self.num_prefix_tokens,
            self.ordered,
            self.training
        )
        return output


class PatchDropoutWithIndices(msnn.Cell):
    """
    Patch Dropout that returns both output and keep indices.
    https://arxiv.org/abs/2212.00794 and https://arxiv.org/pdf/2208.07220
    """

    def __init__(
            self,
            prob: float = 0.5,
            num_prefix_tokens: int = 1,
            ordered: bool = False,
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.num_prefix_tokens = num_prefix_tokens  # exclude CLS token (or other prefix tokens)
        self.ordered = ordered

    def construct(self, x: ms.Tensor) -> Tuple[ms.Tensor, Optional[ms.Tensor]]:
        return patch_dropout_forward(
            x,
            self.prob,
            self.num_prefix_tokens,
            self.ordered,
            self.training
        )
