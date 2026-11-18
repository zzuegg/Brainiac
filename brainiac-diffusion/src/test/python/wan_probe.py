# Compares a Wan transformer in Brainiac with diffusers block by block, on a GPU box.
#   python wan_probe.py inputs <inputs.safetensors> [--frames 3 --height 60 --width 104 --channels 16 --text 4096]
#       random latents, a timestep and text states (a prompt's worth of tokens, then zeros)
#   python wan_probe.py reference <diffusers repo or folder> <inputs.safetensors> <ref.safetensors>
#       runs diffusers' WanTransformer3DModel in bfloat16 and stores every block's output
#   ./gradlew :brainiac-examples:run --args="wanprobe <model folder> inputs.safetensors ours.safetensors"
#       (with -Dbrainiac.wan.residual32=false to keep diffusers' bfloat16 residual stream)
#   python wan_probe.py compare <ref.safetensors> <ours.safetensors> [<more.safetensors> ...]
#       prints how far each block's output is from the reference, relative to its size
import sys, torch
from safetensors.torch import save_file, load_file

cmd = sys.argv[1]
if cmd == "inputs":
    args = sys.argv[3:]
    opt = {args[i]: int(args[i + 1]) for i in range(0, len(args), 2)}
    f, h, w = opt.get("--frames", 3), opt.get("--height", 60), opt.get("--width", 104)
    c, text = opt.get("--channels", 16), opt.get("--text", 4096)
    g = torch.Generator().manual_seed(0)
    context = torch.zeros(1, 512, text)
    context[:, :24] = torch.randn(1, 24, text, generator=g) * 0.2
    save_file({"latents": torch.randn(1, c, f, h, w, generator=g), "timestep": torch.tensor([999.0]),
               "context": context}, sys.argv[2])
elif cmd == "reference":
    from diffusers import WanTransformer3DModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = WanTransformer3DModel.from_pretrained(sys.argv[2], subfolder="transformer",
                                                  torch_dtype=torch.bfloat16).to(dev).eval()
    x = load_file(sys.argv[3])
    out = {}

    def keep(name):
        return lambda m, inp, o: out.__setitem__(name, (o[0] if isinstance(o, tuple) else o).float().cpu())

    model.patch_embedding.register_forward_hook(
        lambda m, inp, o: out.__setitem__("embedded", o.flatten(2).transpose(1, 2).float().cpu()))
    for i, b in enumerate(model.blocks):
        b.register_forward_hook(keep(f"block.{i}"))
    with torch.no_grad():
        r = model(hidden_states=x["latents"].to(dev, torch.bfloat16), timestep=x["timestep"].to(dev),
                  encoder_hidden_states=x["context"].to(dev, torch.bfloat16), return_dict=False)[0]
    out["out"] = r.float().cpu()
    save_file({k: v.contiguous() for k, v in out.items()}, sys.argv[4])
elif cmd == "compare":
    ref = load_file(sys.argv[2])
    runs = [(p, load_file(p)) for p in sys.argv[3:]]
    names = ["embedded"] + sorted((k for k in ref if k.startswith("block.")), key=lambda k: int(k[6:])) + ["out"]
    print("%-10s" % "" + "".join("%24s" % p[-24:] for p, _ in runs))
    for k in names:
        row = "%-10s" % k
        for _, run in runs:
            if k not in run:
                row += "%24s" % "-"
                continue
            a, b = ref[k].flatten(), run[k].reshape(ref[k].shape).flatten()
            rel = ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
            worst = (a - b).abs().max().item()
            row += "%14.4f %9.3g" % (rel, worst)
        print(row)
    print("columns: relative L2 difference, largest absolute difference")
else:
    raise SystemExit("unknown command " + cmd)
