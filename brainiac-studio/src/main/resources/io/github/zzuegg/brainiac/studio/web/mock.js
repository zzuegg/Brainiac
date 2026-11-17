// Example data for the design preview. The real studio fills the same shapes from the
// server's /api endpoints; nothing here is the user's own data.
window.STUDIO_MOCK = {
  capabilities: [
    { key: "text.chat", cap: "text", providers: 3, what: "Chat with tool calls and streaming", source: "brainiac-llm" },
    { key: "text.complete", cap: "text", providers: 3, what: "Raw completion, beam search, speculative decoding", source: "brainiac-llm" },
    { key: "text.embed", cap: "text", providers: 1, what: "Sentence vectors from any loaded model", source: "brainiac-llm" },
    { key: "image.generate", cap: "image", providers: 1, what: "Diffusion and flow-matching samplers", source: "nn.diffusion", note: "No pretrained text-to-image weights yet" },
    { key: "speech.synthesize", cap: "speech", providers: 0, what: "Text to speech with voices", source: "planned", missing: true },
    { key: "speech.transcribe", cap: "speech", providers: 0, what: "Speech to text", source: "planned", missing: true },
    { key: "vision.describe", cap: "vision", providers: 0, what: "Describe images and video frames", source: "planned", missing: true },
    { key: "model.build", cap: "train", providers: 31, what: "Layer blocks with shape checking", source: "brainiac-nn" },
    { key: "model.train", cap: "train", providers: 9, what: "Optimizers, schedules, checkpoints", source: "brainiac-train" },
    { key: "model.finetune", cap: "train", providers: 2, what: "LoRA and QLoRA on Llama-family models", source: "brainiac-llm" },
    { key: "rl.train", cap: "train", providers: 7, what: "PPO, DQN, A2C, REINFORCE, Q-learning", source: "brainiac-rl" },
    { key: "evolve.search", cap: "train", providers: 1, what: "Neuroevolution of wiring and activations", source: "brainiac-evolve" },
    { key: "agent.run", cap: "agent", providers: 1, what: "Orchestrator that plans and spawns sub-agents", source: "studio" },
    { key: "experimental.*", cap: "agent", providers: 12, what: "Blocks found in brainiac.experimental", source: "brainiac-experiments" },
  ],

  models: [
    { name: "Qwen2.5 7B Instruct", cap: ["text"], fmt: "GGUF Q4_K_M", size: "4.7 GB", ctx: "32k", device: "Vulkan", state: "loaded", src: "models/qwen2.5-7b-instruct-q4_k_m.gguf", tps: "41 tok/s" },
    { name: "Qwen2.5 0.5B Instruct", cap: ["text"], fmt: "safetensors bf16", size: "988 MB", ctx: "32k", device: "Vulkan", state: "draft", src: "models/Qwen2.5-0.5B-Instruct", tps: "draft for speculative decoding" },
    { name: "Mistral 7B Instruct v0.3", cap: ["text"], fmt: "GGUF Q8_0", size: "7.7 GB", ctx: "32k", device: "—", state: "on disk", src: "models/mistral-7b-instruct-v0.3.Q8_0.gguf", tps: "" },
    { name: "SmolLM2 1.7B", cap: ["text"], fmt: "safetensors bf16", size: "3.4 GB", ctx: "8k", device: "—", state: "on disk", src: "models/SmolLM2-1.7B-Instruct", tps: "" },
    { name: "thumb-dit-64", cap: ["image"], fmt: ".brainiac", size: "212 MB", ctx: "64×64", device: "Vulkan", state: "loaded", src: "runs/thumb-dit-64/best.brainiac", tps: "trained here" },
    { name: "OpenAI speech", cap: ["speech"], fmt: "remote API", size: "—", ctx: "—", device: "—", state: "remote", src: "openai · voices listed live", tps: "" },
  ],

  threads: [
    { group: "Today", items: [
      { id: "t1", title: "Voiceover for the builder demo", sub: "3 agents · 1 waiting on you", when: "now", cap: "agent", active: true },
      { id: "t1a", title: "Script writer", sub: "done · 6 lines", when: "2m", cap: "text", child: true },
      { id: "t1b", title: "Voiceover", sub: "voicing line 3 of 6", when: "now", cap: "speech", child: true },
      { id: "t1c", title: "Thumbnail", sub: "needs approval", when: "1m", cap: "image", child: true, attention: true },
      { id: "t2", title: "Why does LoRA rank 8 diverge?", sub: "Qwen2.5 7B · 14 messages", when: "11:20", cap: "text" },
    ]},
    { group: "Yesterday", items: [
      { id: "t3", title: "CIFAR-10 ResNet sweep", sub: "Training · 4 runs compared", when: "Mon", cap: "train" },
      { id: "t4", title: "Explain speculative decoding", sub: "Mistral 7B · branched 2×", when: "Mon", cap: "text" },
      { id: "t5", title: "Racing cars PPO tuning", sub: "RL · 2 agents", when: "Mon", cap: "agent" },
    ]},
    { group: "Earlier", items: [
      { id: "t6", title: "Thumbnail style exploration", sub: "thumb-dit-64 · 48 images", when: "Nov 12", cap: "image" },
      { id: "t7", title: "GGUF k-quant memory budget", sub: "Qwen2.5 7B · 6 messages", when: "Nov 10", cap: "text" },
    ]},
  ],

  // Each sub-agent sees only the tools of its own medium. The UI shows exactly those.
  agents: [
    { id: "script", name: "Script writer", cap: "text", model: "Qwen2.5 7B Instruct", status: "done", doing: "Wrote 6 lines timed to the cuts (112 words, 38.4 s)",
      tokens: "1.2k", time: "14 s", progress: 1,
      tools: ["text.chat", "video.scenes (read)", "file.write"] },
    { id: "voice", name: "Voiceover", cap: "speech", model: "OpenAI speech (remote)", status: "running", doing: "Voicing line 3 of 6 · “Drag a block onto the canvas…”",
      tokens: "—", time: "22 s", progress: .42,
      tools: ["speech.synthesize", "speech.retake", "timeline.align", "video.play"] },
    { id: "thumb", name: "Thumbnail", cap: "image", model: "thumb-dit-64", status: "waiting", doing: "Wants to render a batch of 4 at 50 steps (≈ 9 s on GPU)",
      tokens: "—", time: "31 s", progress: .7,
      tools: ["image.generate", "image.vary", "image.upscale", "image.inpaint"] },
  ],

  script: [
    { t: 0.0, d: 5.6, text: "Every model starts as an idea and a blank canvas." },
    { t: 6.0, d: 6.2, text: "Drag a block onto the canvas, and Brainiac checks its shape against everything around it." },
    { t: 12.8, d: 6.0, text: "Connect a convolution to a dense layer, and it tells you why that cannot work, and how to fix it." },
    { t: 19.4, d: 6.4, text: "When the graph is valid, train it on your GPU and watch the loss fall in real time." },
    { t: 26.4, d: 5.8, text: "Then talk to it, generate with it, or hand it to an agent over MCP." },
    { t: 32.8, d: 5.6, text: "Brainiac Studio. Models you can see into." },
  ],

  voices: [
    { name: "Mara", desc: "Calm narrator · warm, low-mid", tags: ["narration", "en"], hue: "var(--cap-speech)", pick: true },
    { name: "Teo", desc: "Bright explainer · fast, clear", tags: ["tutorial", "en"], hue: "var(--cap-text)" },
    { name: "Ilse", desc: "Documentary · measured, deep", tags: ["documentary", "de"], hue: "var(--cap-vision)" },
    { name: "Ravi", desc: "Conversational · relaxed", tags: ["podcast", "en"], hue: "var(--cap-image)" },
    { name: "Noor", desc: "Newsreader · crisp", tags: ["news", "en"], hue: "var(--cap-train)" },
    { name: "Your voice", desc: "Clone from a 30 s sample", tags: ["clone"], hue: "var(--faint)", clone: true },
  ],

  palette: [
    { group: "Inputs and shape", items: [
      { k: "IN", name: "Input", sig: "→ [C,H,W]", cap: "text" },
      { k: "RS", name: "Reshape", sig: "[*] → [*]", cap: "text" },
      { k: "PM", name: "Permute", sig: "[*] → [*]", cap: "text" },
      { k: "PD", name: "Pad", sig: "[C,H,W]", cap: "text" },
    ]},
    { group: "Convolution", items: [
      { k: "CV", name: "Conv", sig: "[C,H,W] → [F,H',W']", cap: "image" },
      { k: "CT", name: "Conv transposed", sig: "↑ upsample", cap: "image" },
      { k: "PL", name: "Max / avg pool", sig: "[C,H,W]", cap: "image" },
      { k: "GP", name: "Global pool", sig: "[C,H,W] → [C]", cap: "image" },
      { k: "PS", name: "Pixel shuffle", sig: "[C·r²,H,W]", cap: "image" },
    ]},
    { group: "Dense and normalization", items: [
      { k: "DN", name: "Dense", sig: "[N] → [U]", cap: "text" },
      { k: "BN", name: "Batch norm", sig: "same shape", cap: "vision" },
      { k: "LN", name: "Layer norm", sig: "same shape", cap: "vision" },
      { k: "GN", name: "Group norm", sig: "same shape", cap: "vision" },
      { k: "DO", name: "Dropout", sig: "same shape", cap: "vision" },
    ]},
    { group: "Sequence and attention", items: [
      { k: "EM", name: "Embedding", sig: "[T] → [T,D]", cap: "speech" },
      { k: "AT", name: "Multi-head attention", sig: "[T,D] → [T,D]", cap: "speech" },
      { k: "TB", name: "Transformer block", sig: "[T,D] → [T,D]", cap: "speech" },
      { k: "RN", name: "Recurrent (LSTM/GRU)", sig: "[T,D] → [T,H]", cap: "speech" },
    ]},
    { group: "Activations", items: [
      { k: "Ac", name: "21 activations", sig: "ReLU, GELU, SiLU…", cap: "agent" },
      { k: "Mx", name: "Mixed / split", sig: "learned blend", cap: "agent" },
    ]},
    { group: "Experimental", experimental: true, items: [
      { k: "BR", name: "Growing brain", sig: "regions + plasticity", cap: "train" },
      { k: "SD", name: "Switching dense", sig: "picks activation", cap: "train" },
      { k: "x*", name: "brainiac.experimental", sig: "found at startup", cap: "train" },
    ]},
  ],

  // A small residual CNN for 32 × 32 colour images, plus one block the user is wiring wrongly.
  // Socket kinds: image = [C, H, W], vector = [N], seq = [T, D].
  graph: {
    nodes: [
      { id: "in", col: 0, lane: "a", title: "Input", cat: "vision", tag: "CIFAR-10",
        ins: [], p: [["dtype", "float32"]], outs: [{ name: "x", kind: "image", shape: "3 × 32 × 32", on: true }] },
      { id: "c1", col: 1, lane: "a", title: "Conv", cat: "image",
        ins: [{ name: "x", kind: "image", shape: "3 × 32 × 32", on: true }],
        p: [["filters", "32"], ["kernel", "3 × 3"], ["activation", "SiLU"]],
        outs: [{ name: "y", kind: "image", shape: "32 × 32 × 32", on: true }] },
      { id: "c2", col: 2, lane: "b", title: "Conv", cat: "image",
        ins: [{ name: "x", kind: "image", shape: "32 × 32 × 32", on: true }],
        p: [["filters", "32"], ["kernel", "3 × 3"], ["padding", "same"]],
        outs: [{ name: "y", kind: "image", shape: "32 × 32 × 32", on: true }] },
      { id: "add", col: 3, lane: "a", title: "Add", cat: "agent",
        ins: [{ name: "a", kind: "image", shape: "32 × 32 × 32", on: true }, { name: "b", kind: "image", shape: "32 × 32 × 32", on: true }],
        p: [], outs: [{ name: "sum", kind: "image", shape: "32 × 32 × 32", on: true }] },
      { id: "gap", col: 4, lane: "a", title: "Global avg pool", cat: "vision",
        ins: [{ name: "x", kind: "image", shape: "32 × 32 × 32", on: true, state: "target" }],
        p: [["over", "H, W"]], outs: [{ name: "y", kind: "vector", shape: "32", on: true }] },
      { id: "d10", col: 5, lane: "a", title: "Dense", cat: "text", tag: "output",
        ins: [{ name: "x", kind: "vector", shape: "32", on: true }],
        p: [["units", "10"], ["activation", "softmax"]], outs: [{ name: "y", kind: "vector", shape: "10", on: true }] },
      { id: "d128", col: 4, lane: "b", dy: 70, title: "Dense", cat: "text", loose: true,
        ins: [{ name: "x", kind: "vector", shape: "N", state: "refuse" }],
        p: [["units", "128"]], outs: [{ name: "y", kind: "vector", shape: "128" }] },
    ],
    edges: [["in.out0", "c1.in0"], ["c1.out0", "c2.in0"], ["c1.out0", "add.in0"], ["c2.out0", "add.in1"], ["add.out0", "gap.in0"], ["gap.out0", "d10.in0"]],
  },

  runs: [
    { name: "resnet-s · adamw", c: "var(--cap-text)", state: "running", host: "This PC", step: "18.4k / 30k", pct: 61, loss: 0.412, acc: "91.2%", eta: "6 min", speed: "5.1k img/s", pts: [2.3,1.9,1.55,1.3,1.1,.98,.86,.78,.71,.66,.61,.57,.54,.51,.49,.47,.46,.45,.44,.43,.42,.415] },
    { name: "resnet-s · muon", c: "var(--cap-speech)", state: "running", host: "Lab workstation", step: "18.4k / 30k", pct: 61, loss: 0.371, acc: "92.6%", eta: "5 min", speed: "7.8k img/s", pts: [2.3,1.8,1.4,1.15,.95,.83,.74,.66,.6,.55,.51,.48,.46,.44,.42,.41,.4,.39,.385,.38,.375,.371] },
    { name: "resnet-s · lion", c: "var(--cap-image)", state: "stopped early", host: "Lab workstation", step: "9.1k", pct: 30, loss: 0.744, acc: "84.0%", eta: "—", speed: "—", pts: [2.3,2.0,1.7,1.45,1.25,1.1,.98,.9,.84,.8,.77,.755,.748,.744] },
    { name: "resnet-s · adam8bit", c: "var(--cap-vision)", state: "queued", host: "next free", step: "0 / 30k", pct: 0, loss: null, acc: "—", eta: "—", speed: "—", pts: [] },
  ],

  // Machines that run training. Each one runs `brainiac-studio --worker`; this PC is always there.
  hosts: [
    { name: "This PC", addr: "local", state: "busy", devices: [{ name: "RX 9070 XT", util: 78, mem: "9.9 / 16 GB" }, { name: "CPU · 16 threads", util: 22, mem: "20 / 64 GB" }], foot: "1 run · chat model loaded" },
    { name: "Lab workstation", addr: "10.0.0.24:7861", state: "busy", devices: [{ name: "RTX 4090 #0", util: 94, mem: "18.2 / 24 GB" }, { name: "RTX 4090 #1", util: 0, mem: "0.4 / 24 GB" }], foot: "1 run · 1 GPU free" },
    { name: "Laptop", addr: "10.0.0.31:7861", state: "offline", devices: [{ name: "Radeon 780M", util: 0, mem: "— / 8 GB" }], foot: "last seen 2 h ago" },
  ],

  // Remote model providers. Model lists come from each provider's own model-list endpoint.
  providers: [
    { name: "Brainiac (this PC)", key: "local", state: "on", caps: ["text", "image", "train"], models: "6 models on disk", note: "free · private · offline" },
    { name: "Anthropic", key: "claude", state: "on", caps: ["text", "vision"], models: "Claude models · listed live", note: "your API key · billed per token" },
    { name: "OpenAI", key: "openai", state: "on", caps: ["text", "image", "speech", "vision"], models: "GPT, image, speech and transcription models", note: "your API key · billed per use" },
    { name: "Google", key: "google", state: "off", caps: ["text", "image", "speech", "vision", "video"], models: "Gemini, Imagen, Veo and speech models", note: "add an API key to use" },
  ],

  mcpTools: [
    { name: "models.list", desc: "Every model on disk or loaded, with its capabilities", args: "{capability?}" },
    { name: "models.load", desc: "Load a model folder or GGUF file on a device", args: "{path, device?, dtype?}" },
    { name: "chat", desc: "Send messages; routes to the best text model unless one is named", args: "{messages, model?, sampling?}" },
    { name: "image.generate", desc: "Sample images from a loaded diffusion model", args: "{prompt, steps, seed, size}" },
    { name: "graph.validate", desc: "Check a model graph and return shapes or the first error with a fix", args: "{graph}" },
    { name: "graph.build", desc: "Create a model from a validated graph", args: "{graph, seed}" },
    { name: "train.start", desc: "Start a training run and stream metrics", args: "{model, data, optimizer, epochs}" },
    { name: "blocks.describe", desc: "List every layer block and its parameters, found at startup", args: "{}" },
  ],

  mcpLog: [
    ["15:41:08", "cursor", "graph.validate  ok  9 nodes, 412k params"],
    ["15:41:02", "cursor", "blocks.describe  ok  31 blocks (12 experimental)"],
    ["15:39:47", "claude", "chat  qwen2.5-7b  218 tok  5.3 s"],
    ["15:39:12", "claude", "models.list  ok  6 models"],
    ["15:31:55", "claude", "train.start  run resnet-s · muon"],
    ["15:31:40", "claude", "graph.build  ok  seed 42"],
  ],
};
