"""Writes the PyTorch checkpoint fixtures; run from this folder with torch and safetensors installed.

model.bin: torch.save of a state dict with views, offsets, several dtypes and a Parameter.
model-legacy.bin: the same, in the format before PyTorch 1.6 (pickles, then raw storages).
expected.safetensors: the same tensors, contiguous, to compare against.
sharded/: a module's state_dict (which pickles its _metadata) split like Hugging Face's
pytorch_model-0000N-of-00002.bin with pytorch_model.bin.index.json.
"""
import json, os, sys, torch
from collections import OrderedDict
from safetensors.torch import save_file

out = sys.argv[1] if len(sys.argv) > 1 else "."
torch.manual_seed(0)
base = torch.randn(4, 6)
sd = OrderedDict()
sd["a.weight"] = base
sd["a.weight_t"] = base.t()
sd["b.half"] = torch.randn(3, 5).half()
sd["b.bf16"] = torch.randn(2, 7).bfloat16()
sd["c.ids"] = torch.arange(10, dtype=torch.int64) - 3
sd["c.slice"] = base[1:3, 2:5]
sd["d.scalar"] = torch.tensor(2.5)
sd["e.flag"] = torch.tensor([True, False, True])
sd["f.param"] = torch.nn.Parameter(torch.randn(3))
torch.save(sd, os.path.join(out, "model.bin"))
torch.save(sd, os.path.join(out, "model-legacy.bin"), _use_new_zipfile_serialization=False)
save_file({k: v.detach().contiguous() for k, v in sd.items()}, os.path.join(out, "expected.safetensors"))

net = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.LayerNorm(4))
state = net.state_dict()
os.makedirs(os.path.join(out, "sharded"), exist_ok=True)
names = list(state)
shards = {"pytorch_model-00001-of-00002.bin": names[:2], "pytorch_model-00002-of-00002.bin": names[2:]}
weight_map = {}
for file, keys in shards.items():
    part = OrderedDict((k, state[k]) for k in keys)
    part._metadata = state._metadata
    torch.save(part, os.path.join(out, "sharded", file))
    weight_map.update({k: file for k in keys})
with open(os.path.join(out, "sharded", "pytorch_model.bin.index.json"), "w") as f:
    json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(out, "sharded-expected.safetensors"))
