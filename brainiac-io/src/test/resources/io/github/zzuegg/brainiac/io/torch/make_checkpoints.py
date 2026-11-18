# PyTorch checkpoints with every storage type Brainiac reads, a nested dict, a parameter,
# a non-contiguous view and a bare tensor. Run: python make_checkpoints.py <this folder>
import collections, json, sys, torch
out = sys.argv[1]
torch.manual_seed(0)
a = torch.randn(3, 4)
sd = collections.OrderedDict()
sd["layer.weight"] = torch.nn.Parameter(a)
sd["layer.half"] = a.half()
sd["layer.bf16"] = a.bfloat16()
sd["layer.double"] = a.double()
sd["layer.transposed"] = a.t()          # strides (1, 4)
sd["layer.slice"] = a[1:, 2:]           # an offset into the storage
sd["steps"] = torch.tensor([5, -7], dtype=torch.long)
nested = {"model": sd, "meta": {"epoch": 3, "name": "x"}, "list": [torch.ones(2)]}
torch.save(nested, f"{out}/nested.pth")
torch.save(a[0], f"{out}/bare.pt")
json.dump({"a": a.numpy().tolist()}, open(f"{out}/expected.json", "w"))
