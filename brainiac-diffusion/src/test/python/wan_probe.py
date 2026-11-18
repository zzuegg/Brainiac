# Compares a Wan transformer in Brainiac with diffusers block by block, on a GPU box.
#   python wan_probe.py inputs <inputs.safetensors> [--frames 3 --height 60 --width 104 --channels 16 --text 4096]
#       random latents, timestep 999 and text states (a prompt's worth of tokens, then zeros)
#   python wan_probe.py sampled <diffusers repo> <inputs.safetensors> [prompt]
#       runs diffusers' WanPipeline (832x480, 9 frames, 6 steps, seed 42) for its final latents and
#       the prompt's text states, and stores cases c0..c3: those latents noised to sigma 0.9, 0.6,
#       0.3 and 0.1 with the matching timesteps, as the sampler meets them; also its starting noise,
#       final latents (x0), shift, and three decoded frames as <inputs>.diffusers-<i>.png
#   ./gradlew :brainiac-examples:run --args="wanprobe pipeline <model folder> inputs.safetensors <out folder>"
#       samples the same clip from the same noise, prints how far its final latents are from
#       diffusers', and writes both decoded by Brainiac's VAE as PNG frames
#   python wan_probe.py reference <diffusers repo or folder> <inputs.safetensors> <ref.safetensors> [f32]
#       runs diffusers' WanTransformer3DModel in bfloat16 (or float32) and stores every block's output
#   ./gradlew :brainiac-examples:run --args="wanprobe <model folder> inputs.safetensors ours.safetensors"
#       (with -Dbrainiac.wan.residual32=false to keep diffusers' bfloat16 residual stream)
#   python wan_probe.py text <diffusers repo> <text.safetensors> [bf16]
#       encodes a few prompts (Wan's Chinese negative among them) as WanPipeline does, with umT5 on
#       the CPU in float32 (or bfloat16), and stores each prompt, its token ids and its text states
#   ./gradlew :brainiac-examples:run --args="wanprobe text <model folder> text.safetensors [--dtype bf16]"
#       encodes the same prompts on the CPU and prints how far the ids and states are from those
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
    from PIL import Image
    prompt = sys.argv[4] if len(sys.argv) > 4 else "a red fox runs through deep snow"
    pipe = WanPipeline.from_pretrained(sys.argv[2], torch_dtype=torch.bfloat16).to("cuda")
    start = torch.randn(1, 16, 3, 60, 104, generator=torch.Generator().manual_seed(42))
    run = dict(prompt=prompt, height=480, width=832, num_frames=9, num_inference_steps=6, guidance_scale=5.0)
    with torch.no_grad():
        x0 = pipe(**run, latents=start.to("cuda"), output_type="latent").frames.float().cpu()
        video = pipe(**run, latents=start.to("cuda"), output_type="np").frames[0]
        embeds, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False, max_sequence_length=512,
                                       device="cuda")
    for i in [0, len(video) // 2, len(video) - 1]:
        Image.fromarray((video[i] * 255).round().clip(0, 255).astype("uint8")).save(f"{sys.argv[3]}.diffusers-{i}.png")
    context = embeds.float().cpu()
    g = torch.Generator().manual_seed(1)
    noise = torch.randn(x0.shape, generator=g)
    out = {"noise": start, "x0": x0, "shift": torch.tensor([float(pipe.scheduler.config.flow_shift)])}
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
elif cmd == "text":
    import html, re
    from transformers import AutoTokenizer, UMT5EncoderModel
    negative = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
                "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
    prompts = [negative, "a red fox runs through deep snow", ""]
    dtype = torch.bfloat16 if sys.argv[4:] == ["bf16"] else torch.float32
    tok = AutoTokenizer.from_pretrained(sys.argv[2], subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(sys.argv[2], subfolder="text_encoder", torch_dtype=dtype).eval()
    out = {}
    for k, p in enumerate(prompts):
        clean = re.sub(r"\s+", " ", html.unescape(html.unescape(p))).strip()
        t = tok([clean], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        with torch.no_grad():
            h = enc(t.input_ids, t.attention_mask).last_hidden_state.float()
        h[:, n:] = 0
        out[f"t{k}.prompt"] = torch.tensor([float(ord(c)) for c in p] or [32.0])
        out[f"t{k}.ids"] = t.input_ids[0, :n].float()
        out[f"t{k}.states"] = h
        print(f"t{k}: {n} tokens, |states| {h.norm().item():.4g}")
    save_file({k: v.contiguous().clone() for k, v in out.items()}, sys.argv[3])
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
