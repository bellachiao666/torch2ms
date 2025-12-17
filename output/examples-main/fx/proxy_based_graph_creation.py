import mindspore as ms
import mindspore.nn as msnn
import mindspore.ops as msops
import mindspore.mint as mint
from mindspore.mint import nn, ops
# import torch
# from torch.fx import Proxy, Graph, GraphModule


'''
How to Create a Graph Using Proxy Objects Instead of Tracing

It's possible to directly create a Proxy object around a raw Node. This
can be used to create a Graph independently of symbolic tracing.

The following code demonstrates how to use Proxy with a raw Node to
append operations to a fresh Graph. We'll create two parameters (``x``
and ``y``), perform some operations on those parameters, then add
everything we created to the new Graph. We'll then wrap that Graph in
a GraphModule. Doing so creates a runnable instance of ``nn.Module``
where previously-created operations are represented in the Module's
``forward`` function.

By the end of the tutorial, we'll have added the following method to an
empty ``nn.Module`` class.

.. code-block:: python

    def forward(self, x, y):
        cat_1 = torch.cat([x, y]);  x = y = None
        tanh_1 = torch.tanh(cat_1);  cat_1 = None
        neg_1 = torch.neg(tanh_1);  tanh_1 = None
        return neg_1

'''


# Create a graph independently of symbolic tracing
graph = Graph()  # 'torch.fx.Graph' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
tracer = torch.fx.proxy.GraphAppendingTracer(graph)  # 'torch.fx.proxy.GraphAppendingTracer' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;

# Create raw Nodes
raw1 = graph.placeholder('x')
raw2 = graph.placeholder('y')

# Initialize Proxies using the raw Nodes and graph's default tracer
y = Proxy(raw1, tracer)  # 'torch.fx.Proxy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
z = Proxy(raw2, tracer)  # 'torch.fx.Proxy' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
# y = Proxy(raw1)
# z = Proxy(raw2)

# Create other operations using the Proxies `y` and `z`
a = mint.cat([y, z])
b = mint.tanh(a)
c = mint.neg(b)
# By using the graph's own appending tracer to create Proxies,
# notice we can now use n-ary operators on operations without
# multiple tracers being created at run-time (line 52) which leads
# to errors # To try this out for yourself, replace lines 42, 43
# with 44, 45
z = mint.add(b, c)

# Create a new output Node and add it to the Graph. By doing this, the
# Graph will contain all the Nodes we just created (since they're all
# linked to the output Node)
graph.output(c.node)

# Wrap our created Graph in a GraphModule to get a final, runnable
# `nn.Module` instance
mod = GraphModule(msnn.Cell(), graph)  # 'torch.fx.GraphModule' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;
