import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch.fx as fx

# An inverse mapping is one that takes a function f(x) and returns a function g
# such that f(g(x)) == x. For example,since log(exp(x)) == x, exp and log are
# inverses.

invert_mapping = {}
def add_inverse(a, b):
    invert_mapping[a] = b
    invert_mapping[b] = a
inverses = [
    (mint.sin, mint.arcsin),
    (mint.cos, mint.arccos),
    (mint.tan, mint.arctan),
    (mint.exp, mint.log),
]
for a, b in inverses:
    add_inverse(a, b)

# The general strategy is that we walk the graph backwards, transforming each
# node into its inverse. To do so, we swap the outputs and inputs of the
# functions, and then we look up its inverse in `invert_mapping`. Note that
# this transform assumes that all operations take in only one input and return
# one output.
def invert(model: msnn.Cell) -> msnn.Cell:
    fx_model = fx.symbolic_trace(model)  # 'torch.fx.symbolic_trace' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    new_graph = fx.Graph()  # As we're building up a new graph; 'torch.fx.Graph' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
    env = {}
    for node in reversed(fx_model.graph.nodes):
        if node.op == 'call_function':
            # This creates a node in the new graph with the inverse function,
            # and passes `env[node.name]` (i.e. the previous output node) as
            # input.
            new_node = new_graph.call_function(invert_mapping[node.target], (env[node.name],))
            env[node.args[0].name] = new_node
        elif node.op == 'output':
            # We turn the output into an input placeholder
            new_node = new_graph.placeholder(node.name)
            env[node.args[0].name] = new_node
        elif node.op == 'placeholder':
            # We turn the input placeholder into an output
            new_graph.output(env[node.name])
        else:
            raise RuntimeError("Not implemented")

    new_graph.lint()
    return fx.GraphModule(fx_model, new_graph)  # 'torch.fx.GraphModule' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;


def f(x):
    return mint.exp(mint.tan(x))

res = invert(f)
print(res.code)
"""
def forward(self, output):
    log_1 = torch.log(output);  output = None
    arctan_1 = torch.arctan(log_1);  log_1 = None
    return arctan_1
"""
print(f(res((mint.arange(5) + 1))))  # [1., 2., 3., 4, 5.]
