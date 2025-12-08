import torch
from torch import nn
import torch.nn.functional as F

from vit_pytorch.vit import ViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.efficient import ViT as EfficientViT
from mindspore.mint import nn, ops

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class DistillMixin:
    def forward(self, img, distill_token = None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = ops.cat(tensors = (cls_tokens, x), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '1 n d -> b n d', b = b)
            x = ops.cat(tensors = (x, distill_tokens), dim = 1)  # 'torch.cat':没有对应的mindspore参数 'out';

        x = self._attend(x)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens

        return out

class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableT2TViT(DistillMixin, T2TViT):
    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = T2TViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableEfficientViT(DistillMixin, EfficientViT):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = EfficientViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        return self.transformer(x)

# knowledge distillation wrapper

class DistillWrapper(nn.Cell):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5,
        hard = False,
        mlp_layernorm = False
    ):
        super().__init__()
        assert (isinstance(student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT))) , 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(ops.randn(size = 1, generator = 1))  # 'torch.randn':没有对应的mindspore参数 'out';; 'torch.randn':没有对应的mindspore参数 'layout';; 'torch.randn':没有对应的mindspore参数 'device';; 'torch.randn':没有对应的mindspore参数 'requires_grad';; 'torch.randn':没有对应的mindspore参数 'pin_memory';

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape = dim) if mlp_layernorm else nn.Identity(),
            nn.Linear(in_features = dim, out_features = num_classes)
        )  # 'torch.nn.LayerNorm':没有对应的mindspore参数 'device';; 'torch.nn.Linear':没有对应的mindspore参数 'device';

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):

        alpha = default(alpha, self.alpha)
        T = default(temperature, self.temperature)

        with torch.no_grad():
            teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.cross_entropy(student_logits, labels)

        if not self.hard:
            distill_loss = F.kl_div(
                ops.special.log_softmax(input = distill_logits / T, dim = -1),
                nn.functional.softmax(input = teacher_logits / T, dim = -1).detach(),
            reduction = 'batchmean')  # 'torch.nn.functional.softmax':没有对应的mindspore参数 '_stacklevel';
            distill_loss *= T ** 2

        else:
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha
