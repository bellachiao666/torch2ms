import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
""" ADOPT PyTorch Optimizer

ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate: https://arxiv.org/abs/2411.02853

Modified for reduced dependencies on PyTorch internals from original at: https://github.com/iShohei220/adopt

@inproceedings{taniguchi2024adopt,
 author={Taniguchi, Shohei and Harada, Keno and Minegishi, Gouki and Oshima, Yuta and Jeong, Seong Cheol and Nagahara, Go and Iiyama, Tomoshi and Suzuki, Masahiro and Iwasawa, Yusuke and Matsuo, Yutaka},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate},
 year = {2024}
}

References for added functionality:
    Cautious Optimizers: https://arxiv.org/abs/2411.16085
    Why Gradients Rapidly Increase Near the End of Training: https://arxiv.org/abs/2506.02285
"""
from typing import cast, List, Optional, Tuple, Union

# import torch
# from torch import Tensor
# from torch.optim.optimizer import Optimizer

from ._types import ParamsT

__all__ = ["Adopt", "adopt"]

def _view_as_real(params, *state_and_grads):
    for i, p in enumerate(params):
        if torch.is_complex(p):
            params[i] = torch.view_as_real(params[i])  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            for s in state_and_grads:
                s[i] = torch.view_as_real(s[i])  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def _get_scalar_dtype(is_fused=None):
    if is_fused:
        return ms.float32
    return (
        ms.float64 if torch.get_default_dtype() == ms.float64 else ms.float32
    )  # 'torch.get_default_dtype' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def _is_compiling():
    if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'is_compiling'):
        return torch.compiler.is_compiling()  # 'torch.compiler.is_compiling' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    else:
        return False


def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and _is_compiling():
        return x
    else:
        return x.item() if isinstance(x, torch.Tensor) else x


# 'torch.optim.optimizer.Optimizer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
class Adopt(Optimizer):
    """
    ADOPT: Modified Adam Can Converge with Any β2 with the Optimal Rate: https://arxiv.org/abs/2411.02853

    """
    def __init__(
            self,
            params: ParamsT,
            lr: Union[float, ms.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.9999),
            eps: float = 1e-6,
            clip_exp: Optional[float] = 0.333,
            weight_decay: float = 0.0,
            decoupled: bool = False,
            corrected_weight_decay: bool = False,
            *,
            caution: bool = False,
            foreach: Optional[bool] = False,
            maximize: bool = False,
            capturable: bool = False,
            differentiable: bool = False,
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clip_exp=clip_exp,
            decoupled=decoupled,
            corrected_weight_decay=corrected_weight_decay,
            caution=caution,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("clip_exp", None)
            group.setdefault("caution", False)
            group.setdefault("corrected_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        ms.Tensor(
                            step_val, dtype = _get_scalar_dtype())
                        if group["capturable"]
                        else ms.Tensor(step_val, dtype = _get_scalar_dtype())
                    )  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;; 'torch.tensor':没有对应的mindspore参数 'device' (position 2);

    def _init_group(
            self,
            group,
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= torch.is_complex(p)  # 'torch.is_complex' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("ADOPT does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            # Lazy state initialization
            if len(state) == 0:
                # note(crcrpar): [special device hosting for step]
                # Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    mint.zeros((), dtype = _get_scalar_dtype())
                    if group["capturable"]
                    else ms.Tensor(0.0, dtype = _get_scalar_dtype())
                )  # 'torch.zeros':没有对应的mindspore参数 'device' (position 4);; 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;
                # Exponential moving average of gradient values
                state["exp_avg"] = mint.zeros_like(p.grad)  # 'torch.zeros_like':没有对应的mindspore参数 'memory_format' (position 5);
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = mint.zeros_like(p.grad)  # 'torch.zeros_like':没有对应的mindspore参数 'memory_format' (position 5);

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError("`requires_grad` is not supported for `step` in differentiable mode")

            # Foreach without capturable does not support a tensor lr
            if group["foreach"] and torch.is_tensor(group["lr"]) and not group["capturable"]:
                raise RuntimeError("lr as a Tensor is not supported for capturable=False and foreach=True")

            state_steps.append(state["step"])
        return has_complex

    #@_use_grad_for_differentiable  # FIXME internal context mgr, can't use
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[ms.Tensor] = []
            grads: List[ms.Tensor] = []
            exp_avgs: List[ms.Tensor] = []
            exp_avg_sqs: List[ms.Tensor] = []
            state_steps: List[ms.Tensor] = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            adopt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                clip_exp=group["clip_exp"],
                max_lr=self.defaults['lr'] if group['corrected_weight_decay'] else None,
                decoupled=group["decoupled"],
                eps=group["eps"],
                caution=group["caution"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def _single_tensor_adopt(
        params: List[ms.Tensor],
        grads: List[ms.Tensor],
        exp_avgs: List[ms.Tensor],
        exp_avg_sqs: List[ms.Tensor],
        state_steps: List[ms.Tensor],
        grad_scale: Optional[ms.Tensor],
        found_inf: Optional[ms.Tensor],
        *,
        has_complex: bool,
        beta1: float,
        beta2: float,
        lr: Union[float, ms.Tensor],
        weight_decay: float,
        clip_exp: Optional[float],
        max_lr: Optional[float],
        decoupled: bool,
        eps: float,
        caution: bool,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if capturable and not _is_compiling():
            # from torch.optim.optimizer import _get_capturable_supported_devices
            capturable_supported_devices = _get_capturable_supported_devices()  # 'torch.optim.optimizer._get_capturable_supported_devices' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            assert param.device.type == step_t.device.type and param.device.type in capturable_supported_devices,\
                f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        # update step
        step_t += 1

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            if exp_avg is not None:
                exp_avg = torch.view_as_real(exp_avg)  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            if exp_avg_sq is not None:
                exp_avg_sq = torch.view_as_real(exp_avg_sq)  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            param = torch.view_as_real(param)  # 'torch.view_as_real' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        if weight_decay != 0 and not decoupled:
            grad = grad.add(param, alpha=weight_decay)

        step = step_t if capturable or differentiable else _get_value(step_t)
        if step == 1:
            exp_avg_sq.addcmul_(grad, grad.conj())
            continue

        if weight_decay != 0 and decoupled:
            wd_scale = lr ** 2 / max_lr if max_lr is not None else lr
            param.add_(param, alpha=-wd_scale * weight_decay)

        denom = mint.clamp(exp_avg_sq.sqrt(), eps)
        normed_grad = grad.div(denom)

        if clip_exp is not None:
            clip_val = (step - 1) ** clip_exp
            normed_grad.clamp_(-clip_val, clip_val)

        exp_avg.lerp_(normed_grad, 1 - beta1)

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            mask = (exp_avg * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            exp_avg = exp_avg * mask

        param.add_(exp_avg, alpha=-lr)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)


def _multi_tensor_adopt(
        params: List[ms.Tensor],
        grads: List[ms.Tensor],
        exp_avgs: List[ms.Tensor],
        exp_avg_sqs: List[ms.Tensor],
        state_steps: List[ms.Tensor],
        grad_scale: Optional[ms.Tensor],
        found_inf: Optional[ms.Tensor],
        *,
        has_complex: bool,
        beta1: float,
        beta2: float,
        lr: Union[float, ms.Tensor],
        weight_decay: float,
        clip_exp: Optional[float],
        max_lr: Optional[float],
        decoupled: bool,
        eps: float,
        caution: bool,
        maximize: bool,
        capturable: bool,
        differentiable: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if capturable and not _is_compiling():
        # from torch.optim.optimizer import _get_capturable_supported_devices
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )  # 'torch.optim.optimizer._get_capturable_supported_devices' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        assert all(
            p.device.type == step.device.type and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )  # 'torch.optim.optimizer.Optimizer._group_tensors_by_device_and_dtype' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    for (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(List[Tensor], device_params_)
        device_grads = cast(List[Tensor], device_grads_)
        device_exp_avgs = cast(List[Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(List[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(List[Tensor], device_state_steps_)

        # Handle complex parameters
        if has_complex:
            _view_as_real(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs)

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]; 'torch._foreach_neg' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not _is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(device_state_steps, ms.Tensor(1.0), alpha=1.0)  # 'torch.tensor':默认参数名不一致(position 0): PyTorch=data, MindSpore=input_data;; 'torch.tensor':没有对应的mindspore参数 'device' (position 2);; 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        else:
            torch._foreach_add_(device_state_steps, 1)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        if weight_decay != 0 and not decoupled:
            # Re-use the intermediate memory (device_grads) already allocated for maximize
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)  # 'torch._foreach_add' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        if device_state_steps[0] == 1:
            torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads)  # 'torch._foreach_addcmul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            continue

        if weight_decay != 0 and decoupled:
            wd_scale = lr ** 2 / max_lr if max_lr is not None else lr
            torch._foreach_add_(device_params, device_params, alpha=-wd_scale * weight_decay)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)  # 'torch._foreach_sqrt' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_maximum_(exp_avg_sq_sqrt, eps)  # 'torch._foreach_maximum_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        normed_grad = torch._foreach_div(device_grads, exp_avg_sq_sqrt)  # 'torch._foreach_div' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        if clip_exp is not None:
            clip_val = (device_state_steps[0] - 1) ** clip_exp
            torch._foreach_maximum_(normed_grad, -clip_val)  # 'torch._foreach_maximum_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            torch._foreach_minimum_(normed_grad, clip_val)  # 'torch._foreach_minimum_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        torch._foreach_lerp_(device_exp_avgs, normed_grad, 1 - beta1)  # 'torch._foreach_lerp_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        if caution:
            # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
            masks = torch._foreach_mul(device_exp_avgs, device_grads)  # 'torch._foreach_mul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            masks = [(m > 0).to(g.dtype) for m, g in zip(masks, device_grads)]
            mask_scale = [m.mean() for m in masks]
            torch._foreach_maximum_(mask_scale, 1e-3)  # 'torch._foreach_maximum_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            torch._foreach_div_(masks, mask_scale)  # 'torch._foreach_div_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            device_exp_avgs = torch._foreach_mul(device_exp_avgs, masks)  # 'torch._foreach_mul' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        torch._foreach_add_(device_params, device_exp_avgs, alpha=-lr)  # 'torch._foreach_add_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

        torch._foreach_mul_(device_exp_avg_sqs, beta2)  # 'torch._foreach_mul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        torch._foreach_addcmul_(device_exp_avg_sqs, device_grads, device_grads, value=1 - beta2)  # 'torch._foreach_addcmul_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


#@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adopt)  # FIXME internal context mgr, can't use
def adopt(
        params: List[ms.Tensor],
        grads: List[ms.Tensor],
        exp_avgs: List[ms.Tensor],
        exp_avg_sqs: List[ms.Tensor],
        state_steps: List[ms.Tensor],
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        grad_scale: Optional[ms.Tensor] = None,
        found_inf: Optional[ms.Tensor] = None,
        has_complex: bool = False,
        *,
        beta1: float,
        beta2: float,
        lr: Union[float, ms.Tensor],
        weight_decay: float,
        clip_exp: Optional[float],
        max_lr: Optional[float],
        decoupled: bool,
        eps: float,
        caution: bool,
        maximize: bool,
):
    r"""Functional API that performs ADOPT algorithm computation.

    """
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not _is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adopt
    else:
        func = _single_tensor_adopt

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        clip_exp=clip_exp,
        max_lr=max_lr,
        decoupled=decoupled,
        eps=eps,
        caution=caution,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
