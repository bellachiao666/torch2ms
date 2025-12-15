import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
from typing import Dict, Optional, cast

# from einops._torch_specific import apply_for_scriptable_torch

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = "Alex Rogozhnikov"


class Rearrange(RearrangeMixin, msnn.Cell):
    def construct(self, input):
        recipe = self._multirecipe[input.ndim]
        return apply_for_scriptable_torch(recipe, input, reduction_type="rearrange", axes_dims=self._axes_lengths)

    def _apply_recipe(self, x):
        # overriding parent method to prevent it's scripting
        pass


class Reduce(ReduceMixin, msnn.Cell):
    def construct(self, input):
        recipe = self._multirecipe[input.ndim]
        return apply_for_scriptable_torch(recipe, input, reduction_type=self.reduction, axes_dims=self._axes_lengths)

    def _apply_recipe(self, x):
        # overriding parent method to prevent it's scripting
        pass


class EinMix(_EinmixMixin, msnn.Cell):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = ms.Parameter(
            mint.zeros(weight_shape).uniform_(-weight_bound, weight_bound), requires_grad=True
        )
        if bias_shape is not None:
            self.bias = ms.Parameter(
                mint.zeros(bias_shape).uniform_(-bias_bound, bias_bound), requires_grad=True
            )
        else:
            self.bias = None

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern: Optional[str],
        pre_reshape_lengths: Optional[Dict],
        post_reshape_pattern: Optional[str],
        post_reshape_lengths: Optional[Dict],
    ):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))

    def construct(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = mint.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
