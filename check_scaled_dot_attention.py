try:
    from mindspore.text.modules.attentions import ScaledDotAttention
    print("ScaledDotAttention import success:", ScaledDotAttention)
except Exception as err:
    print("ScaledDotAttention import failed:", repr(err))
