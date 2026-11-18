"""Writes small ONNX models and their onnxruntime outputs for OnnxModelTest.

Run from this folder: python -I make_onnx.py
"""
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, helper, numpy_helper
from safetensors.numpy import save_file

torch.manual_seed(0)
rng = np.random.default_rng(0)
OUT = "onnx"


def finish(name, path, feeds):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, feeds)
    data = {}
    for k, v in feeds.items():
        data["in." + k] = np.ascontiguousarray(v.astype(np.float32))
    for o, v in zip(sess.get_outputs(), outs):
        data["out." + o.name] = np.ascontiguousarray(np.asarray(v).astype(np.float32))
    save_file(data, f"{OUT}/{name}.safetensors")


def export(name, model, args, names, dynamic=None):
    model.eval()
    path = f"{OUT}/{name}.onnx"
    torch.onnx.export(model, args, path, input_names=names, output_names=["y"], opset_version=17,
                      dynamic_axes=dynamic, dynamo=False)
    feeds = {n: a.numpy() for n, a in zip(names, args)}
    finish(name, path, feeds)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 16)
        self.b = nn.Linear(16, 5)

    def forward(self, x):
        return F.softmax(self.b(F.gelu(self.a(x))), dim=-1)


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.c2 = nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=2)
        self.up = nn.ConvTranspose2d(8, 4, 2, stride=2)
        self.fc = nn.Linear(4 * 8 * 8, 6)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn(self.c1(x))), 2)            # 8x8
        x = F.leaky_relu(self.c2(x), 0.1)                            # 4x4
        x = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        x = F.interpolate(x, scale_factor=2, mode="nearest")         # 8x8
        x = F.hardswish(self.up(x))                                  # 16x16
        x = F.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)
        return self.fc(torch.flatten(x, 1))


class Block(nn.Module):
    def __init__(self, vocab=40, dim=32, heads=4):
        super().__init__()
        self.heads = heads
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(64, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        b, t = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(t))
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(x.shape[-1], dim=-1)
        q = q.view(b, t, self.heads, -1).transpose(1, 2)
        k = k.view(b, t, self.heads, -1).transpose(1, 2)
        v = v.view(b, t, self.heads, -1).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
        mask = torch.ones(t, t, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, -1)
        x = x + self.proj(y)
        x = x + self.down(F.gelu(self.up(self.ln2(x)), approximate="tanh"))
        return F.log_softmax(self.head(x), -1)


def ops():
    """A hand-built graph over many operators at once."""
    f = TensorProto.FLOAT
    i64 = TensorProto.INT64
    c = lambda name, arr: numpy_helper.from_array(np.asarray(arr), name)
    nodes = [
        helper.make_node("Slice", ["x", "s0", "s1", "s2", "s3"], ["sl"]),            # reversed rows, every 2nd col
        helper.make_node("Pad", ["x", "pads"], ["pd"], mode="reflect"),
        helper.make_node("ReduceMax", ["pd", "ax12"], ["rm"], keepdims=0),
        helper.make_node("CumSum", ["x", "one"], ["cs"], reverse=1),
        helper.make_node("ArgMax", ["x"], ["am"], axis=2, keepdims=0),
        helper.make_node("Cast", ["am"], ["amf"], to=f),
        helper.make_node("Greater", ["x", "zero"], ["gt"]),
        helper.make_node("Where", ["gt", "x", "cs"], ["wh"]),
        helper.make_node("Tile", ["sl", "reps"], ["tl"]),
        helper.make_node("Einsum", ["x", "x"], ["es"], equation="bij,bkj->bik"),
        helper.make_node("Trilu", ["es"], ["tr"], upper=0),
        helper.make_node("Split", ["x", "splits"], ["sp0", "sp1"], axis=2),
        helper.make_node("Clip", ["sp1", "lo", "hi"], ["cl"]),
        helper.make_node("Shape", ["x"], ["shp"]),
        helper.make_node("ConstantOfShape", ["shp"], ["cos"], value=helper.make_tensor("v", f, [1], [0.5])),
        helper.make_node("Add", ["cos", "x"], ["ad"]),
        helper.make_node("Mod", ["ad", "modv"], ["md"], fmod=1),
        helper.make_node("HardSigmoid", ["md"], ["hs"], alpha=0.3, beta=0.4),
        helper.make_node("InstanceNormalization", ["x4", "sc", "bi"], ["inn"], epsilon=1e-5),
        helper.make_node("LayerNormalization", ["x", "lnw", "lnb"], ["lnn"], axis=-1, epsilon=1e-5),
        helper.make_node("GatherElements", ["x", "gi"], ["ge"], axis=2),
        helper.make_node("Range", ["r0", "r1", "r2"], ["rg"]),
        helper.make_node("Cast", ["rg"], ["rgf"], to=f),
        helper.make_node("Expand", ["rgf", "eshape"], ["ex"]),
        helper.make_node("Resize", ["x4", "", "rscale"], ["rs"], mode="linear"),
        helper.make_node("ReduceSum", ["rs", "ax23"], ["rss"], keepdims=1),
        helper.make_node("Unsqueeze", ["x", "zero64"], ["ux"]),
        helper.make_node("Squeeze", ["ux", "zero64"], ["sx"]),
        helper.make_node("Neg", ["sx"], ["ng"]),
        helper.make_node("Pow", ["ng", "two"], ["pw"]),
    ]
    inits = [
        c("s0", np.array([-1, 0], np.int64)), c("s1", np.array([-100, 100], np.int64)),
        c("s2", np.array([1, 2], np.int64)), c("s3", np.array([-1, 2], np.int64)),
        c("pads", np.array([0, 1, 2, 0, 1, 1], np.int64)), c("ax12", np.array([1, 2], np.int64)),
        c("one", np.array(1, np.int64)), c("zero", np.array(0, np.float32)),
        c("reps", np.array([1, 2, 1], np.int64)), c("splits", np.array([2, 4], np.int64)),
        c("lo", np.array(-0.5, np.float32)), c("hi", np.array(0.7, np.float32)),
        c("modv", np.array(0.75, np.float32)),
        c("sc", rng.standard_normal(3).astype(np.float32)), c("bi", rng.standard_normal(3).astype(np.float32)),
        c("lnw", rng.standard_normal(6).astype(np.float32)), c("lnb", rng.standard_normal(6).astype(np.float32)),
        c("gi", rng.integers(-6, 6, (2, 4, 3)).astype(np.int64)),
        c("r0", np.array(2, np.int64)), c("r1", np.array(11, np.int64)), c("r2", np.array(3, np.int64)),
        c("eshape", np.array([2, 1], np.int64)), c("rscale", np.array([1, 1, 2, 1.5], np.float32)),
        c("ax23", np.array([2, 3], np.int64)), c("zero64", np.array([0], np.int64)),
        c("two", np.array(2.0, np.float32)),
    ]
    outs = ["sl", "pd", "rm", "cs", "amf", "wh", "tl", "tr", "sp0", "cl", "hs", "inn", "lnn", "ge", "ex",
            "rss", "pw"]
    g = helper.make_graph(nodes, "ops",
                          [helper.make_tensor_value_info("x", f, [2, 4, 6]),
                           helper.make_tensor_value_info("x4", f, [2, 3, 4, 4])],
                          [helper.make_tensor_value_info(o, f, None) for o in outs], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 8
    path = f"{OUT}/ops.onnx"
    onnx.save(m, path)
    finish("ops", path, {"x": rng.standard_normal((2, 4, 6)).astype(np.float32),
                         "x4": rng.standard_normal((2, 3, 4, 4)).astype(np.float32)})


export("mlp", Mlp(), (torch.randn(3, 8),), ["x"])
cnn = Cnn()
with torch.no_grad():
    cnn.bn.running_mean.uniform_(-0.5, 0.5)
    cnn.bn.running_var.uniform_(0.5, 1.5)
export("cnn", cnn, (torch.randn(2, 3, 16, 16),), ["x"])
export("block", Block(), (torch.randint(0, 40, (2, 7)),), ["ids"], dynamic={"ids": {0: "b", 1: "t"}})
ops()
