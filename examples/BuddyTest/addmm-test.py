import os
import torch
from torch._inductor.decomposition import decompositions as inductor_decomp

from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops.gpu import ops_registry as gpu_ops_registry
from model import AddMMModule

model = AddMMModule()
model = model.eval()

# Initialize the dynamo compiler.
dynamo_compiler = DynamoCompiler(
    primary_registry=gpu_ops_registry,
    aot_autograd_decomposition=inductor_decomp,
)

input_data = torch.randn(4,2)  
# print(type(input_data))

with torch.no_grad():
    graphs = dynamo_compiler.importer(model, input_data)

graph = graphs[0]
graph.lower_to_top_level_ir()
path_prefix = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(path_prefix, "addmm.mlir"), "w") as module_file:
    print(graph._imported_module, file=module_file)
