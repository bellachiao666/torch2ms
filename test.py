import mindspore as ms
from mindspore import Tensor
import numpy as np
import output.timm as timm

def test_forward():
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=1000,
    )
    model.set_train(False)

    x = Tensor(np.random.randn(1, 3, 224, 224), ms.float32)
    y = model(x)

    print("Output shape:", y.shape)
    assert y.shape == (1, 1000)

if __name__ == "__main__":
    test_forward()
