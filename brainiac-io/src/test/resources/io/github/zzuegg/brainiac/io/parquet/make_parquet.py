"""Writes the Parquet fixtures with pyarrow; run from this folder. expected.jsonl holds the rows,
with binary values as {"$bytes": hex}."""
import json, random
import pyarrow as pa, pyarrow.parquet as pq

random.seed(0)
n = 300
rows = []
for i in range(n):
    rows.append({
        "id": i * 1_000_003 - 7,
        "small": None if i % 7 == 0 else i % 100 - 50,
        "score": None if i % 11 == 0 else i / 8.0 - 3,
        "ratio": float(i % 13) / 4,
        "flag": None if i % 5 == 0 else i % 3 == 0,
        "text": random.choice(["alpha", "beta", "gamma", "", "ünïcødé"]) + str(i % 4),
        "blob": bytes([i % 256, (i * 7) % 256]) if i % 9 else None,
        "tags": None if i % 10 == 3 else [f"t{j}" for j in range(i % 4)],
        "messages": [{"role": "user" if j % 2 == 0 else "assistant", "content": None if (i + j) % 6 == 5 else f"m{i}-{j}"}
                     for j in range(i % 3)],
        "matrix": [[i + j * k for k in range(j % 3)] for j in range(i % 4)],
        "meta": {"lang": "en" if i % 2 else "de", "len": i % 50} if i % 4 else None,
        "attrs": {f"k{j}": j * i for j in range(i % 3)},
    })
schema = pa.schema([
    ("id", pa.int64()), ("small", pa.int32()), ("score", pa.float64()), ("ratio", pa.float32()),
    ("flag", pa.bool_()), ("text", pa.string()), ("blob", pa.binary()), ("tags", pa.list_(pa.string())),
    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("matrix", pa.list_(pa.list_(pa.int64()))),
    ("meta", pa.struct([("lang", pa.string()), ("len", pa.int32())])),
    ("attrs", pa.map_(pa.string(), pa.int64())),
])
table = pa.Table.from_pylist(rows, schema=schema)
variants = {
    "snappy.parquet": dict(compression="snappy", row_group_size=128),
    "zstd-v2.parquet": dict(compression="zstd", data_page_version="2.0", row_group_size=1000),
    "gzip.parquet": dict(compression="gzip", use_dictionary=False),
    "lz4.parquet": dict(compression="lz4", data_page_size=512),
    "plain.parquet": dict(compression="none", use_dictionary=False, data_page_version="2.0"),
    "delta.parquet": dict(compression="zstd", use_dictionary=False, column_encoding={
        "id": "DELTA_BINARY_PACKED", "small": "DELTA_BINARY_PACKED", "text": "DELTA_BYTE_ARRAY",
        "blob": "DELTA_LENGTH_BYTE_ARRAY", "ratio": "BYTE_STREAM_SPLIT", "score": "BYTE_STREAM_SPLIT"}),
}
for name, kw in variants.items():
    pq.write_table(table, name, **kw)

def plain(v):
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    if isinstance(v, dict):
        return {k: plain(x) for k, x in v.items()}
    if isinstance(v, list):
        if v and isinstance(v[0], tuple):   # a map
            return {k: plain(x) for k, x in v}
        return [plain(x) for x in v]
    return v

with open("expected.jsonl", "w") as f:
    for r in table.to_pylist():
        r["attrs"] = dict(r["attrs"])
        f.write(json.dumps(plain(r), ensure_ascii=False) + "\n")
