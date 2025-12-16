import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" NAdamW Optimizer

Based on simplified algorithm in https://github.com/mlcommons/algorithmic-efficiency/tree/main/baselines/nadamw

Added multi-tensor (foreach) path.

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285
"""
import math
from typing import List, Optional, Tuple

# import torch
# from torch import Tensor

from ._types import ParamsT


# Modified from github.com/pytorch/pytorch/blob/v1.12.1/torch/optim/adamw.py.
# 'torch.optim.Optimizer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class NAdamW(torch.optim.Optimizer):
    """ Implements NAdamW algorithm.

    See Table 1 in https://arxiv.org/abs/1910.05446 for the implementation of
    the NAdam algorithm (there is also a comment in the code which highlights
    the only difference of NAdamW and AdamW).

    For further details regarding the algorithm we refer to
        - Decoupled Weight Decay Regularization: https://arxiv.org/abs/1711.05101
        - On the Convergence of Adam and Beyond: https://openreview.net/forum?id=ryQu7f-RZ

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr: learning rate
        betas: coefficients used for computing running averages of gradient and its square
        eps: term added to the denominator to improve numerical stability
        weight_decay: weight decay coefficient
        caution: enable caution
        corrected_weight_decay: apply corrected weight decay (lr**2 / max_lr)
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
            caution: bool = False,
            corrected_weight_decay: bool = False,
            maximize: bool = False,
            foreach: Optional[bool] = None,
            capturable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            caution=caution,
            corrected_weight_decay=corrected_weight_decay,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])  # 'torch.is_tensor' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if not step_is_tensor:
            for s in state_values:
                s['step'] = ms.Tensor(float(s['step']))  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;
        for group in self.param_groups:
            group.setdefault('caution', False)
            group.setdefault('corrected_weight_decay', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

            Args:
              closure (callable, optional): A closure that reevaluates the model
                  and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('NAdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = ms.Tensor(0.)  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;
                    # Exponential moving average of gradient values
                    state['exp_avg'] = mint.zeros_like(p)  # 'torch.zeros_like':没有对应的mindspore参数 'memory_format' (position 5);
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = mint.zeros_like(p)  # 'torch.zeros_like':没有对应的mindspore参数 'memory_format' (position 5);

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                state_steps.append(state['step'])

            nadamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                caution=group['caution'],
                maximize=group['maximize'],
                capturable=group['capturable'],
                max_lr=self.defaults['lr'] if group['corrected_weight_decay'] else None,
            )

        return loss


def nadamw(
        params: List[ms.Tensor],
        grads: List[ms.Tensor],
        exp_avgs: List[ms.Tensor],
        exp_avg_sqs: List[ms.Tensor],
        state_steps: List[ms.Tensor],
        foreach: Optional[bool] = None,
        capturable: bool = False,
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        caution: bool,
        maximize: bool,
        max_lr: Optional[float],
) -> None:
    r"""Functional API that performs NAdamW algorithm computation.
      See NAdamW class for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            'API has changed, `state_steps` argument must contain a list of' +
            ' singleton tensors')

    if foreach is None:
        try:
            # cannot do foreach if this overload doesn't exist when caution enabled
            foreach = not caution or 'Scalar' in torch.ops.aten._foreach_maximum_.overloads()  # 'torch.ops.aten._foreach_maximum_.overloads' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        except:
            foreach = False

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_nadamw
    else:
        func = _single_tensor_nadamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        caution=caution,
        maximize=maximize,
        capturable=capturable,
        max_lr=max_lr,
    )


def _single_tensor_nadamw(
        params: List[ms.Tensor],
        grads: List[ms.Tensor],
        exp_avgs: List[ms.Tensor],
        exp_avg_sqs: List[ms.Tensor],
        state_steps: List[ms.Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        caution: bool,
        maximize: bool,
        capturable: bool,
        max_lr: Optional[float],
):

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Update step.
        step_t += 1

        # Perform stepweight decay.
        wd_scale = lr if max_lr is None else lr ** 2 / max_lr
        param.mul_(1. - wd_scale * weight_decay)

        # Decay the first and second moment running average coefficient.
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - mint.pow(beta1, step)
            bias_correction2 = 1 - mint.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            # Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            exp_avg = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)

            denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            if caution:
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                # FIXME not 100% sure if this remains capturable?
                mask = (exp_avg * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                exp_avg.mul_(mask)

            param.addcdiv_(exp_avg, denom)
        else:
            step = step_t.item()
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)

            # Apply Nesterov. Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            exp_avg = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1)
            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            if caution:
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                mask = (exp_avg * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                exp_avg.mul_(mask)

            param.addcdiv_(exp_avg, denom, value=-step_size)


def _multi_tensor_nadamw(
        params: List[ms.Tensor],
        grads: List[ms.Tensor],
        exp_avgs: List[ms.Tensor],
        exp_avg_sqs: List[ms.Tensor],
        state_steps: List[ms.Tensor],
        *,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        caution: bool,
        maximize: bool,
        capturable: bool,
        max_lr: Optional[float],
):
    if len(params) == 0:
        return

    if capturable:
        assert all(
            p.is_cuda and step.is_cuda for p, step in zip(params, state_steps)
        ), "If capturable=True, params and state_steps must be CUDA tensors."

    if maximize:
        grads = torch._foreach_neg(tuple(grads))  # type: ignore[assignment]; 'torch._foreach_neg' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    grads = [torch.view_as_real(x) if torch.is_complex(x) else x for x in grads]  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.is_complex' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    exp_avgs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avgs]  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.is_complex' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    exp_avg_sqs = [torch.view_as_real(x) if torch.is_complex(x) else x for x in exp_avg_sqs]  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.is_complex' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    params = [torch.view_as_real(x) if torch.is_complex(x) else x for x in params]  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch.is_complex' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    # update steps
    torch._foreach_add_(state_steps, 1)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    # Perform stepweight decay
    wd_scale = lr if max_lr is None else lr ** 2 / max_lr
    torch._foreach_mul_(params, 1 -  wd_scale * weight_decay)  # 'torch._foreach_mul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    # Decay the first and second moment running average coefficient
    torch._foreach_mul_(exp_avgs, beta1)  # 'torch._foreach_mul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    torch._foreach_mul_(exp_avg_sqs, beta2)  # 'torch._foreach_mul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)  # 'torch._foreach_addcmul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    if capturable:
        # TODO: use foreach_pow if/when foreach_pow is added
        bias_correction1 = [mint.pow(beta1, step) for step in state_steps]
        bias_correction2 = [mint.pow(beta2, step) for step in state_steps]
        # foreach_sub doesn't allow a scalar as the first arg
        torch._foreach_sub_(bias_correction1, 1)  # 'torch._foreach_sub_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_sub_(bias_correction2, 1)  # 'torch._foreach_sub_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_neg_(bias_correction1)  # 'torch._foreach_neg_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_neg_(bias_correction2)  # 'torch._foreach_neg_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        # foreach_div doesn't allow a scalar as the first arg
        step_size = torch._foreach_div(bias_correction1, lr)  # 'torch._foreach_div' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_reciprocal_(step_size)  # 'torch._foreach_reciprocal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_neg_(step_size)  # 'torch._foreach_neg_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        bias_correction2_sqrt = torch._foreach_sqrt(bias_correction2)  # 'torch._foreach_sqrt' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        # Only difference between NAdamW and AdamW in this implementation.
        # The official PyTorch implementation of NAdam uses a different algorithm.
        exp_avgs = torch._foreach_mul(exp_avgs, beta1)  # 'torch._foreach_mul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)  # 'torch._foreach_sqrt' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_div_(
            exp_avg_sq_sqrt,
            torch._foreach_mul(bias_correction2_sqrt, step_size)
        )  # 'torch._foreach_mul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;; 'torch._foreach_div_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        eps_over_step_size = torch._foreach_div(step_size, eps)  # 'torch._foreach_div' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_reciprocal_(eps_over_step_size)  # 'torch._foreach_reciprocal_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        denom = torch._foreach_add(exp_avg_sq_sqrt, eps_over_step_size)  # 'torch._foreach_add' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            masks = torch._foreach_mul(exp_avgs, grads)  # 'torch._foreach_mul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            masks = [(m > 0).to(g.dtype) for m, g in zip(masks, grads)]  # capturable?
            mask_scale = [m.mean() for m in masks]
            torch._foreach_maximum_(mask_scale, 1e-3)  # 'torch._foreach_maximum_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            #torch._foreach_clamp_min_(mask_scale, 1e-3)
            torch._foreach_div_(masks, mask_scale)  # 'torch._foreach_div_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            torch._foreach_mul_(exp_avgs, masks)  # 'torch._foreach_mul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        torch._foreach_addcdiv_(params, exp_avgs, denom)  # 'torch._foreach_addcdiv_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    else:
        bias_correction1 = [1 - beta1 ** step.item() for step in state_steps]
        bias_correction2 = [1 - beta2 ** step.item() for step in state_steps]

        step_size = [(lr / bc) * -1 for bc in bias_correction1]

        bias_correction2_sqrt = [math.sqrt(bc) for bc in bias_correction2]

        # Apply Nesterov. Only difference between NAdamW and AdamW in this implementation.
        # The official PyTorch implementation of NAdam uses a different algorithm.
        exp_avgs = torch._foreach_mul(exp_avgs, beta1)  # 'torch._foreach_mul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        exp_avg_sq_sqrt = torch._foreach_sqrt(exp_avg_sqs)  # 'torch._foreach_sqrt' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)  # 'torch._foreach_div_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        denom = torch._foreach_add(exp_avg_sq_sqrt, eps)  # 'torch._foreach_add' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            masks = torch._foreach_mul(exp_avgs, grads)  # 'torch._foreach_mul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            masks = [(m > 0).to(g.dtype) for m, g in zip(masks, grads)]
            mask_scale = [m.mean() for m in masks]
            torch._foreach_maximum_(mask_scale, 1e-3)  # 'torch._foreach_maximum_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            #torch._foreach_clamp_min_(mask_scale, 1e-3)
            torch._foreach_div_(masks, mask_scale)  # 'torch._foreach_div_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            torch._foreach_mul_(exp_avgs, masks)  # 'torch._foreach_mul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        torch._foreach_addcdiv_(params, exp_avgs, denom, step_size)  # 'torch._foreach_addcdiv_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
