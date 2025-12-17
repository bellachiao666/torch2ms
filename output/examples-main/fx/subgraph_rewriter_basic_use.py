import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# from torch.fx import symbolic_trace, replace_pattern


'''
How to Use the FX Subgraph Rewriter

For easy subgraph rewriting, FX exposes the utility function:

    replace_pattern(gm : GraphModule,
                    pattern : Callable,
                    replacement : Callable)
                    -> None

`replace_pattern` matches all possible non-overlapping sets of operators
and their data dependencies (`pattern`) in the Graph of a GraphModule
(`gm`), then replaces each of these matched subgraphs with another
subgraph (`replacement).

The docstring for `replace_pattern` (located in `subgraph_rewriter.py`)
gives an in-depth explanation as to how `pattern` and `replacement`
should be specified, what happens during pattern matching, and other
important technical details. This tutorial, therefore, is only meant to
give an overview as to the FX Subgraph Rewriter's basic functionality.
Let's go rewrite a Graph!
'''

# Sample module
class M(msnn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, w1, w2):
        val1 = mint.neg(w1)
        m1 = mint.cat([val1, w2]).sum()
        val2 = mint.neg(w1)
        m2 = mint.cat([val2, w2]).sum()
        return x + mint.max(m1) + mint.max(m2)

# Symbolically trace an instance of `M`
traced = symbolic_trace(M())  # 'torch.fx.symbolic_trace' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

# Define the pattern. The FX Subgraph Rewriter will match all
# non-overlapping instances of the pattern in the larger graph.
# Note that Pattern-matching is done based on data dependencies,
# not Node names. Even though we're operating on Nodes named `a1` and
# `a2` instead of `w1` and `w2`, the pattern is still a valid match
# for the two instances of `torch.cat([w1, w2]).sum()` above. Only
# operations that contribute to the single output value of the pattern
# are considered
def pattern(a1, a2):
    val1 = mint.neg(a1)
    return mint.cat([val1, a2]).sum()

# Define the replacement (same rules as the pattern)
def replacement(w1, w2):
    return mint.stack([w1, w2])

# Replace `pattern` with `replacement` in `traced`
replace_pattern(traced, pattern, replacement)  # 'torch.fx.replace_pattern' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

# After calling `replace_pattern`, the generated code is:
'''
def forward(self, x, w1, w2):
    stack = torch.stack([w1, w2])
    max_1 = torch.max(stack);  stack = None
    add = x + max_1;  x = max_1 = None
    stack_1 = torch.stack([w1, w2]);  w1 = w2 = None
    max_2 = torch.max(stack_1);  stack_1 = None
    add_1 = add + max_2;  add = max_2 = None
    return add_1
'''
