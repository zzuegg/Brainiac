# Compares a Wan transformer in Brainiac with diffusers block by block, on a GPU box.
#   python wan_probe.py inputs <inputs.safetensors> [--frames 3 --height 60 --width 104 --channels 16 --text 4096]
#       random latents, timestep 999 and text states (a prompt's worth of tokens, then zeros)
#   python wan_probe.py sampled <diffusers repo> <inputs.safetensors> [prompt]
#       runs diffusers' WanPipeline (832x480, 9 frames, 6 steps, seed 42) for its final latents and
#       the prompt's text states, and stores cases c0..c3: those latents noised to sigma 0.9, 0.6,
#       0.3 and 0.1 with the matching timesteps, as the sampler meets them
#   python wan_probe.py reference <diffusers repo or folder> <inputs.safetensors> <ref.safetensors> [f32]
#       runs diffusers' WanTransformer3DModel in bfloat16 (or float32) and stores every block's output
#   ./gradlew :brainiac-examples:run --args="wanprobe <model folder> inputs.safetensors ours.safetensors"
#       (with -Dbrainiac.wan.residual32=false to keep diffusers' bfloat16 residual stream)
#   python wan_probe.py compare <ref.safetensors> <ours.safetensors> [<more.safetensors> ...]
#       prints how far each block's output is from the reference, relative to its size
import sys, torch
from safetensors.torch import save_file, load_file


def cases(keys):
    return sorted({k[:-len("latents")] for k in keys if k.endswith("latents")})


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
elif cmd == "sampled":
    from diffusers import WanPipeline
    prompt = sys.argv[4] if len(sys.argv) > 4 else "a red fox runs through deep snow"
    pipe = WanPipeline.from_pretrained(sys.argv[2], torch_dtype=torch.bfloat16).to("cuda")
    with torch.no_grad():
        x0 = pipe(prompt=prompt, height=480, width=832, num_frames=9, num_inference_steps=6, guidance_scale=5.0,
                  generator=torch.Generator("cuda").manual_seed(42), output_type="latent").frames.float().cpu()
        embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False, max_sequence_length=512,
                                       device="cuda")
    context = embeds.float().cpu()
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(x0.shape, generator=g)
    out = {}
    for i, sigma in enumerate([0.9, 0.6, 0.3, 0.1]):
        out[f"c{i}.latents"] = (1 - sigma) * x0 + sigma * noise
        out[f"c{i}.timestep"] = torch.tensor([sigma * 1000.0])
        out[f"c{i}.context"] = context
    save_file({k: v.contiguous().clone() for k, v in out.items()}, sys.argv[3])
elif cmd == "reference":
    from diffusers import WanTransformer3DModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if sys.argv[5:] == ["f32"] else torch.bfloat16
    model = WanTransformer3DModel.from_pretrained(sys.argv[2], subfolder="transformer",
                                                  torch_dtype=dtype).to(dev).eval()
    x = load_file(sys.argv[3])
    out = {}
    prefix = [""]

    def keep(name):
        return lambda m, inp, o: out.__setitem__(prefix[0] + name,
                                                 (o[0] if isinstance(o, tuple) else o).float().cpu())

    model.patch_embedding.register_forward_hook(
        lambda m, inp, o: out.__setitem__(prefix[0] + "embedded", o.flatten(2).transpose(1, 2).float().cpu()))
    for i, b in enumerate(model.blocks):
        b.register_forward_hook(keep(f"block.{i}"))
    for c in cases(x):
        prefix[0] = c
        with torch.no_grad():
            r = model(hidden_states=x[c + "latents"].to(dev, dtype), timestep=x[c + "timestep"].to(dev),
                      encoder_hidden_states=x[c + "context"].to(dev, dtype), return_dict=False)[0]
        out[c + "out"] = r.float().cpu()
    save_file({k: v.contiguous() for k, v in out.items()}, sys.argv[4])
elif cmd == "compare":
    ref = load_file(sys.argv[2])
    runs = [(p, load_file(p)) for p in sys.argv[3:]]
    for c in sorted({k[:-len("out")] for k in ref if k.endswith("out")}):
        blocks = sorted((k for k in ref if k.startswith(c + "block.")), key=lambda k: int(k.split(".")[-1]))
        print("%-14s" % c + "".join("%24s" % p[-24:] for p, _ in runs))
        for k in [c + "embedded"] + blocks + [c + "out"]:
            row = "%-14s" % k
            for _, run in runs:
                if k not in run:
                    row += "%24s" % "-"
                    continue
                a, b = ref[k].flatten(), run[k].reshape(ref[k].shape).flatten()
                rel = ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
                worst = (a - b).abs().max().item()
                row += "%14.4f %9.3g" % (rel, worst)
            print(row)
        print()
    print("columns: relative L2 difference, largest absolute difference")
else:
    raise SystemExit("unknown command " + cmd)
