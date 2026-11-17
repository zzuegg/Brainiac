import numpy as np, gguf, sys
from gguf import GGMLQuantizationType as T
import gguf.quants as q
out=sys.argv[1]
rng=np.random.default_rng(7)
w=gguf.GGUFWriter(out, arch="llama")
w.add_uint32("test.u32", 42); w.add_float32("test.f32", 0.25); w.add_string("test.str","héllo")
w.add_bool("test.bool", True); w.add_array("test.arr", ["a","b"]); w.add_array("test.ints",[1,2,3])
w.add_uint64("test.u64", 1<<40); w.add_int8("test.i8", -3)
fields={T.Q4_0:[0],T.Q4_1:[0,2],T.Q5_0:[0],T.Q5_1:[0,2],T.Q8_0:[0],
        T.Q2_K:[80,82],T.Q3_K:[108],T.Q4_K:[0,2],T.Q5_K:[0,2],T.Q6_K:[208]}
for t,offs in fields.items():
    bs,ts=gguf.GGML_QUANT_SIZES[t]
    rows=3; cols=bs*2 if bs==256 else bs*4
    nb=rows*cols//bs
    raw=rng.integers(0,256,size=(nb,ts),dtype=np.uint8)
    for o in offs:
        vals=(rng.uniform(-0.1,0.1,size=nb)).astype(np.float16).view(np.uint8).reshape(nb,2)
        raw[:,o:o+2]=vals
    raw=raw.reshape(rows, cols//bs*ts)
    exp=q.dequantize(raw, t).astype(np.float32).reshape(rows,cols)
    w.add_tensor("q."+t.name, raw, raw_dtype=t)
    w.add_tensor("expected."+t.name, exp)
# Q8_K: f32 d then 256 int8 then 16 int16
bs,ts=gguf.GGML_QUANT_SIZES[T.Q8_K]
raw=rng.integers(0,256,size=(2,ts),dtype=np.uint8)
raw[:,0:4]=np.array([0.01,-0.02],dtype=np.float32).view(np.uint8).reshape(2,4)
d=raw[:,0:4].copy().view(np.float32); exp=(raw[:,4:260].view(np.int8).astype(np.float32)*d).reshape(1,512)
w.add_tensor("q.Q8_K", raw.reshape(1,2*ts), raw_dtype=T.Q8_K); w.add_tensor("expected.Q8_K", exp)
x=rng.uniform(-1,1,size=(2,3,4)).astype(np.float32)
w.add_tensor("plain.f32", x); w.add_tensor("plain.f16", x.astype(np.float16))
w.add_tensor("plain.i32", np.arange(5,dtype=np.int32))
bf=q.quantize(x, T.BF16); w.add_tensor("plain.bf16", bf, raw_dtype=T.BF16)
w.add_tensor("expected.bf16", q.dequantize(bf,T.BF16).astype(np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
