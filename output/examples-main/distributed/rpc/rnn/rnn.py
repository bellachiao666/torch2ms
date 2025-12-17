import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch
# import torch.nn as nn
# import torch.distributed.rpc as rpc
# from torch.distributed.rpc import RRef


def _call_method(method, rref, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)  # 存在 *args/**kwargs，未转换，需手动确认参数映射;


def _remote_method(method, rref, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    return rpc.rpc_sync(
        rref.owner(),
        _call_method,
        args=[method, rref] + list(args),
        kwargs=kwargs
    )  # 'torch.distributed.rpc.rpc_sync' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def _parameter_rrefs(module):
    r"""
    Create one RRef for each parameter in the given local module, and return a
    list of RRefs.
    """
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))  # 'torch.distributed.rpc.RRef' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    return param_rrefs


class EmbeddingTable(msnn.Cell):
    r"""
    Encoding layers of the RNNModel
    """
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            self.encoder = self.encoder.to(device)
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)  # 'torch.nn.init.uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, input):
        # 'torch.accelerator.is_available' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator()  # 'torch.accelerator.current_accelerator' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
            input = input.to(device)
        return self.drop(self.encoder(input)).cpu()


class Decoder(msnn.Cell):
    r"""
    Decoding layers of the RNNModel
    """
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        nn.init.zeros_(self.decoder.bias)  # 'torch.nn.init.zeros_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)  # 'torch.nn.init.uniform_' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, output):
        return self.decoder(self.drop(output))


class RNNModel(msnn.Cell):
    r"""
    A distributed RNN model which puts embedding table and decoder parameters on
    a remote parameter server, and locally holds parameters for the LSTM module.
    The structure of the RNN model is borrowed from the word language model
    example. See https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()

        # setup embedding table remotely
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, args=(ntoken, ninp, dropout))  # 'torch.distributed.rpc.remote' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # setup LSTM locally
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)  # 'torch.nn.LSTM' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
        # setup decoder remotely
        self.decoder_rref = rpc.remote(ps, Decoder, args=(ntoken, nhid, dropout))  # 'torch.distributed.rpc.remote' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

    def construct(self, input, hidden):
        # pass input to the remote embedding table and fetch emb tensor back
        emb = _remote_method(EmbeddingTable.forward, self.emb_table_rref, input)
        output, hidden = self.rnn(emb, hidden)
        # pass output to the remote decoder and get the decoded output back
        decoded = _remote_method(Decoder.forward, self.decoder_rref, output)
        return decoded, hidden

    def parameter_rrefs(self):
        remote_params = []
        # get RRefs of embedding table
        remote_params.extend(_remote_method(_parameter_rrefs, self.emb_table_rref))
        # create RRefs for local parameters
        remote_params.extend(_parameter_rrefs(self.rnn))
        # get RRefs of decoder
        remote_params.extend(_remote_method(_parameter_rrefs, self.decoder_rref))
        return remote_params
