# Brainiac

A neural network library written from scratch in Java, with zero runtime dependencies.

Brainiac aims to cover everything from a tiny fully connected network to a full training
suite behind a small, easy public API. Every calculation goes through a pluggable
`Backend`, so the same model can later run on the GPU.

> Status: on top of tensors, autograd, dense networks, training and data loading, there
> are dropout, normalization, L2, learning-rate schedules, graph models, saving and
> loading, custom ops, 21 activations, convolutional networks for 1-d, 2-d and 3-d
> data, and weight quantization (post-training and quantization-aware training).

## Requirements

- JDK 27 installed (the build uses a Gradle toolchain; Gradle itself runs on JDK 17+)
- Nothing else: the Gradle wrapper is included

## Build

```sh
./gradlew build                      # compile, test, javadoc
./gradlew :brainiac-examples:run --args=iris   # run an example
```

## Modules

| Module | Contents |
| --- | --- |
| `brainiac-tensor` | Shapes, tensors (float32, float64, float16, bfloat16), tensor math, and the `Backend` SPI |
| `brainiac-backend-cpu` | Default backend, plain Java on the CPU |
| `brainiac-backend-vulkan` | Optional GPU backend: Vulkan compute shaders through LWJGL (its only dependency) |
| `brainiac-backend-cuda` | Optional NVIDIA GPU backend: the CUDA driver, NVRTC and cuBLAS called through the Foreign Function and Memory API (no dependencies) |
| `brainiac-backend-hip` | Optional AMD GPU backend (ROCm/HIP, RDNA cards): the HIP runtime, hipRTC and rocBLAS called through the Foreign Function and Memory API (no dependencies) |
| `brainiac-autograd` | Reverse-mode automatic differentiation |
| `brainiac-nn` | Layers, activations, losses, initializers, models, the growing `Brain` |
| `brainiac-train` | Optimizers (Adam/AdamW, Lion, Adafactor, Muon, parameter groups), training loops, checkpoints, EMA, metrics, listeners, `Plasticity` |
| `brainiac-data` | Datasets, batching, splitting, CSV loading, preprocessing |
| `brainiac-io` | File formats: JSON, memory-mapped safetensors (single or sharded) and GGUF weights, weight-name mapping, pretrained tokenizers |
| `brainiac-llm` | Pretrained language models (Llama, Llama 3, Qwen2 from Hugging Face or GGUF files), a KV cache, sampling, streaming, batched generation and beam search |
| `brainiac-audio` | Speech: WAV files, Whisper speech-to-text (any length, timestamps, language detection, translation) and text-to-speech with VITS / MMS-TTS, Kokoro (voices and blends), F5-TTS (voice cloning) and Orpheus (emotion tags), from published weights, behind `SpeechToText` and `TextToSpeech` |
| `brainiac-diffusion` | Image generation with pretrained Stable Diffusion 1.x, 2.x and XL (diffusers folders or single `.safetensors` checkpoints): CLIP tokenizer and text encoders, UNet, VAE, Euler / Euler ancestral / DPM++ 2M (Karras) samplers, classifier-free guidance; FLUX.1 dev and schnell (T5-XXL, MMDiT transformer, flow matching, 8- or 4-bit weights); text- and image-to-video with Wan 2.1 and 2.2 (umT5, CLIP vision, video DiT, causal 3D VAE, UniPC) |
| `brainiac-rl` | Reinforcement learning: environments, replay buffers, Q-learning, SARSA, REINFORCE, A2C, PPO, DQN |
| `brainiac-evolve` | Neuroevolution: a genetic algorithm that searches wiring, neuron counts, per-neuron activations and hyperparameters |
| `brainiac` | Facade: depend on this one module to get the whole API |
| `brainiac-examples` | Runnable examples |
| `brainiac-benchmarks` | Benchmark harness for kernels and training speed |
| `brainiac-experiments` | Experimental research ideas (memory, context, new architectures, learning rules, evolution, generative toys, reasoning, agents, unconventional computing, interpretability). Untested; its API may change at any time |
| `brainiac-studio` | Local web app: chat, generation, a shape-checked model builder, training, and agents that use the studio's tools. JDK HTTP server, no dependencies |
| `brainiac-cloud` | Renting GPUs on Vast.ai: an API client and a launcher that runs jobs or a studio worker there (see `cloud/vast/README.md`) |

Shared build configuration lives in `build-logic/` as Gradle convention plugins
(`brainiac.java-library`, `brainiac.java-application`). Each module has a
`module-info.java`; only public API packages are exported.

## Example

```java
Dataset data = Dataset.of(
        new float[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
        new float[][] {{0}, {1}, {1}, {0}});

Model model = Model.sequential()
        .input(2)
        .dense(8, Activation.TANH)
        .dense(1, Activation.SIGMOID)
        .seed(42)
        .build();

Trainer.of(model)
        .loss(Loss.MSE)
        .optimizer(Optimizer.adam(0.05f))
        .batchSize(4)
        .epochs(2_000)
        .listener(Progress.every(250))
        .fit(data);

float[] out = model.predict(1, 0);   // ~[0.999]
```

Loading CSV data, splitting and validating:

```java
Dataset iris = Dataset.csv(Path.of("iris.csv"))
        .features("sepal_length", "sepal_width", "petal_length", "petal_width")
        .label("species").oneHot()
        .load();
Dataset.Split split = iris.shuffle(42).split(0.8);

Trainer.of(model)
        .loss(Loss.CROSS_ENTROPY)
        .metric(Metric.ACCURACY)
        .validation(split.test())
        .earlyStopping(30)
        .fit(split.train());
```

Regularization and learning-rate schedules:

```java
Model model = Model.sequential()
        .input(4)
        .dense(64).batchNorm().activation(Activation.RELU)
        .dropout(0.2f)
        .dense(3, Activation.SOFTMAX)
        .build();

Trainer.of(model)
        .optimizer(Optimizer.adam(0.01f))
        .l2(1e-4f)
        .schedule(LearningRateSchedule.warmup(5, LearningRateSchedule.cosine(100, 1e-4f)))
        .epochs(100)
        .fit(data);
```

Graphs with skip connections and branches:

```java
Node input = Node.input(4);
Node hidden = input.dense(32, Activation.RELU);
Node block = hidden.dense(32, Activation.RELU).dense(32).add(hidden).activation(Activation.RELU);
Node output = block.concat(hidden).dense(3, Activation.SOFTMAX);
Model model = Model.graph(input, output).seed(42).build();
```

Built-in activations: `IDENTITY`, `RELU`, `RELU6`, `leakyRelu(slope)`, `elu(alpha)`, `SELU`,
`GELU`, `GELU_TANH`, `SILU`, `MISH`, `SOFTPLUS`, `SOFTSIGN`, `SIGMOID`, `HARD_SIGMOID`,
`HARD_SWISH`, `TANH`, `HARD_TANH`, `softmax(axis)`/`SOFTMAX`, `logSoftmax(axis)`/`LOG_SOFTMAX`,
and the learnable `prelu()` and `swish()`. Each one picks a suitable weight init.

Different activations per group of units, or a learned blend per unit:

```java
model.dense(32, Activation.split(Activation.RELU, Activation.TANH))          // 16 + 16 units
model.dense(4, Activation.split(new int[] {1, 1, 2}, RELU, SIGMOID, GELU))   // any sizes, even per unit
model.dense(32, Activation.mix(Activation.RELU, Activation.TANH, Activation.SILU))  // learns the mix
```

Custom and learnable activations (autograd derives the gradient):

```java
Activation swish = Activation.named("swish", x -> x.mul(x.sigmoid()));
Activation scaledTanh = Activation.withParameters("scaled-tanh",
        width -> List.of(Tensor.ones(Shape.of(width))),   // one learnable scale per unit
        (x, p) -> x.tanh().mul(p.get(0)));

Model model = Model.sequential()
        .input(4)
        .dense(32, Activation.prelu())      // learnable negative slope per unit
        .dense(32, Activation.swish())      // learnable beta per unit
        .dense(16, scaledTanh)
        .dense(3, Activation.SOFTMAX)
        .build();
```

A custom operation with a hand-written gradient (for speed, functions autograd cannot
express, or tricks like straight-through estimators), checked numerically:

```java
Op cube = new Op() {
    public Tensor forward(Tensor... in) { return in[0].mul(in[0]).mul(in[0]); }
    public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
        return new Tensor[] {grad.mul(in[0].square()).mul(3)};
    }
};
assert GradCheck.check(cube, x).passed();
Activation cubeActivation = Activation.named("cube", v -> Var.apply(cube, v));
```

### Optimizers and long training runs

Optimizers: `sgd`, `momentum`, `adam`, `adamW`, `lion`, `adafactor` (factored second
moment, optionally with relative steps as T5 trained) and `muon` (orthogonalized momentum for
matrices). Each matches PyTorch's or transformers' implementation on the same gradients.
`adam8bit` keeps Adam's two averages as 8-bit levels with a scale per block of up to 256
values (`Tensor.quantized`, on the GPU without leaving it): a quarter of the state memory,
training along nearly the same path.
Parameter groups give different parameters different optimizers or settings:

```java
Optimizer opt = Optimizer.groups(Optimizer.adamW(3e-4f, 0.1f),   // biases, norms, embeddings
        Optimizer.Group.matrices(Optimizer.muon(0.02f)));           // weight matrices
```

`TrainingLoop` trains step by step on losses you compute, for models that do not fit
`Trainer`'s epochs, such as language models. It accumulates gradients over micro-batches,
clips them, schedules the learning rate per step, keeps an exponential moving average of
the weights, frees each step's memory, and saves checkpoints that resume exactly:

```java
TrainingLoop loop = TrainingLoop.of(model.namedParameters(), opt)
        .accumulate(8)                                    // 8 micro-batches per step
        .clipGradientNorm(1)
        .schedule(LearningRateSchedule.warmup(200, LearningRateSchedule.cosine(20_000, 3e-5f)))
        .ema(0.999f);
loop.resumeIfPresent(Path.of("checkpoints/run1"));        // weights, optimizer, averages, step
while (loop.step() < 20_000) {
    loop.microBatch(() -> lossOn(nextBatch()));
    if (loop.stepped() && loop.step() % 1000 == 0) {
        loop.save(Path.of("checkpoints/run1"));
    }
}
loop.ema().swap();                                        // use the averaged weights
```

Corpora larger than memory are kept as `TokenShards`: memory-mapped shards of 16- or
32-bit token ids (nanoGPT's and llm.c's `.bin` format), read as training windows in order
(with a position to save in a checkpoint) or at random:

```java
try (TokenShards.Writer w = TokenShards.writer(Path.of("data/web"), 100_000_000, tok.size())) {
    for (String doc : documents) w.add(tok.encode(doc), endOfText);
}
TokenShards data = TokenShards.open(Path.of("data/web"));
int[][] windows = data.next(16, 1024);   // 16 windows of 1024 inputs plus the next token
loop.save(dir, Map.of("position", data.position()));   // later: data.seek(...)
```

`Recompute.apply(block, x, block.parameters())` runs a block without keeping what it
computes inside and recomputes it in the backward pass (activation checkpointing), so deep
models train in much less memory for about a third more compute.

## Images: convolution and pooling

Models take samples of any shape. Images are channels first (`input(3, 32, 32)`), and the
number of kernel sizes picks 1-d, 2-d, 3-d (or higher) convolutions. Flat dataset rows are
reshaped to the input shape automatically.

```java
Model cnn = Model.sequential()
        .input(1, 28, 28)
        .add(Conv.of(16, 3, 3).padding(Padding.SAME).bias(false))
        .batchNorm()
        .activation(Activation.RELU)
        .maxPool(2, 2)
        .conv2d(32, 3, Activation.RELU)
        .globalAvgPool()
        .dense(10, Activation.SOFTMAX)
        .build();
```

Every option lives on a spec, used with `add(...)` on a builder or a `Node`:

```java
Conv.of(64, 3, 3)
    .stride(2)                        // or per dimension: stride(2, 1)
    .padding(Padding.SAME)            // VALID, SAME, FULL, of(1), explicit(before, after)
    .padMode(PadMode.REFLECT)         // CONSTANT (padValue), REFLECT, SYMMETRIC, REPLICATE, CIRCULAR
    .dilation(2)                      // atrous
    .groups(4)                        // grouped
    .bias(false).activation(Activation.prelu()).init(Init.HE);
Conv.depthwise(3, 3).multiplier(2);                         // depthwise
builder.separableConv(64, Activation.RELU, 3, 3);           // depthwise + pointwise
Conv.transposed(32, 4, 4).stride(2).padding(1);              // upsampling, outputPadding(...)
Conv.of(8, 5).stride(2);  Conv.of(8, 3, 3, 3);               // 1-d and 3-d

Pool.max(3, 3).stride(2).padding(1).ceilMode(true).dilation(1);
Pool.avg(2, 2).countPadding(false);  Pool.lp(2, 3, 3);  Pool.min(2, 2);  Pool.sum(2, 2);
GlobalPool.avg();  GlobalPool.max();  AdaptivePool.avg(7, 7);  AdaptivePool.max(1, 1);
```

More image layers: `Pad` (any mode, negative amounts crop), `Resize` (nearest,
bilinear/trilinear, bicubic, by factor or to a size, `alignCorners`), `PixelShuffle`
(sub-pixel up/down), `Reshape`/`flatten()`, `Permute`, `spatialDropout(rate)`,
`BatchNorm` per channel, `GroupNorm`, `GroupNorm.instance(c)`, `LayerNorm` and
`LayerNorm.rms(n)`. Graphs can `concat` channels, `add` and `multiply` feature maps.

The same math is available as functions on `Var` in `Functional` (`conv`,
`convTranspose`, `pool`, `globalPool`, `adaptivePool`, `pad`, `resize`, `pixelShuffle`),
built on two backend operations, `Tensor.im2col` and `Tensor.col2im`, plus batched
`matmul`, `gather` and `scatterAdd`.

MNIST (IDX files, plain or gzipped) loads with
`Dataset.idx(images, labels)`; `./gradlew :brainiac-examples:run --args=mnist` downloads it
and trains a small CNN to about 99% test accuracy.

Saving and loading:

```java
model.save(Path.of("iris.brainiac"));
Model restored = Model.load(Path.of("iris.brainiac"));

// custom activations and layers load once registered (learned values are restored automatically)
Activation.register(scaledTanh);
Layer.register("my-layer", MyLayer.class, new LayerCodec<>() {
    public Config write(MyLayer l) { return Config.builder().put("width", l.width()).build(); }
    public MyLayer read(Config c) { return new MyLayer(c.getInt("width")); }
});
```

Tensors and gradients directly:

```java
Var w = Var.param(Tensor.uniform(Shape.of(2, 8), -1, 1, new SplittableRandom(1)));
Var loss = Var.of(x).matmul(w).tanh().sub(y).square().mean();
loss.backward();
Tensor dw = w.grad();
```

Run the examples:

```sh
./gradlew :brainiac-examples:run --args=xor
./gradlew :brainiac-examples:run --args=spiral
./gradlew :brainiac-examples:run --args=iris
./gradlew :brainiac-examples:run --args="mnist data/mnist 3"   # directory, epochs
```

## Words: embeddings

`Embedding` turns whole-number token ids `[steps]` into learned vectors
`[steps, dimension]`. Only the rows of ids in a batch get gradient.

```java
Embedding.of(5000, 64)
    .paddingIndex(0)                  // id 0 is padding: always zero, never trained
    .maxNorm(3)                       // rescale longer looked-up vectors
    .scaleBySqrtDimension(true)       // as transformer inputs usually are
    .init(Init.XAVIER);               // default N(0, 1)
Embedding.pretrained(gloveVectors).frozen();   // [vocabulary, dimension]; kept fixed, still saved
builder.embedding(5000, 64);                   // shorthand on Model.Builder and Node
embedding.nearest(tokenId, 5);                 // most similar tokens (cosine), to inspect what was learned
```

Sentences of different lengths are padded to one length. `PaddingMask` turns the ids into a
1/0 mask, and masked layers take it as an extra input so padding is ignored:

```java
Node ids = Node.input(12);
Node mask = ids.layer(new PaddingMask(0));
Node out = ids.add(Embedding.of(vocabulary, 32).paddingIndex(0))
        .positionalEncoding()
        .merge(TransformerBlock.of(4).keyMask(true), mask)      // attention skips padded steps
        .merge(MaskedPool.of(SequencePool.Kind.MEAN), mask)      // MEAN, SUM, MAX, FIRST, LAST of real steps
        .dense(2, Activation.SOFTMAX);
Model model = Model.graph(ids, out).build();
```

`MultiHeadAttention.of(h).keyMask(true)` works the same way.
`./gradlew :brainiac-examples:run --args=sentiment` learns word vectors from scratch to read
made-up reviews such as "it is great but the plot is dull". A bag of words (the mean of the
vectors, ignoring order) gets about 87%, and a bi-LSTM and a masked transformer get 100%,
because they read "not" and "but" in order. It then prints each word's nearest neighbours:
"good" lands next to "wonderful, excellent, fun", and "awful" next to "boring, dull, bad, poor".

## Text: tokenizers and generation

A `Tokenizer` turns text into token ids and back. It is a table learned once by counting,
not a network. Every kind except `char` starts from the 256 bytes, so no text is unknown:

```java
Tokenizer t = Tokenizer.train("bpe:1024", text);   // char, byte, word:N, bpe:N, wordpiece:N, unigram:N
int[] ids = t.encode("What light through yonder window breaks?");
// [What][ light][ through][ y][onder][ wind][ow][ break][s][?] with bpe:2048 on Shakespeare
String back = t.decode(ids);
t.save(path);  Tokenizer.load(path);
```

For many classes, such as a vocabulary of tokens, targets can be class ids `[steps]`
instead of one-hot rows `[steps, classes]`: cross-entropy and accuracy accept both, and
the one-hot rows are never stored.

`KvCache` generates with a causal transformer one token at a time. Each attention layer
keeps the keys and values of earlier tokens, so a new token is one step through the
network instead of a pass over the whole window; past the window, the oldest are dropped:

```java
KvCache cache = KvCache.of(model);
Tensor p = cache.append(tokenizer.encode("ROMEO:\n"));   // next-token probabilities
p = cache.append(nextToken);
```

`bench shakespeare --tokenizers char,bpe:1024,unigram:1024` trains the same small GPT on
Shakespeare with each tokenizer and reports the loss in nats per character, so they
compare fairly (char-level nanoGPT reaches about 1.47).

## Pretrained weights: safetensors and GGUF

`WeightSource.open` reads a `.safetensors` file, a sharded model's `*.index.json`, or a
directory downloaded from Hugging Face. Files are memory-mapped, so opening a 40 GB model
takes almost no memory, and on the GPU backends each tensor's bytes go straight from the
mapping to the device without passing through the Java heap. Conversions such as
bfloat16 to float32 run on the device. `NameMap` renames weights with regex rules.

```java
NameMap names = NameMap.create()
        .rule("model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight", "block$1.query")
        .drop(".*rotary_emb\\.inv_freq");
try (WeightSource w = WeightSource.open(Path.of("Qwen2.5-0.5B")).renamed(names)) {
    for (String name : w.names()) {
        System.out.println(name + " " + w.format(name) + " " + w.shape(name));
    }
    Tensor q0 = w.tensor("block0.query", Device.gpu(), DType.FLOAT16);
}
SafeTensors.write(Path.of("mine.safetensors"), Map.of("w", q0));   // each tensor in its own type
```

GGUF files (llama.cpp's format) open the same way. Every common quantized format
(`Q4_0` to `Q8_0`, and the k-quants `Q2_K` to `Q8_K`) decodes to float, checked against
llama.cpp's own gguf-py. `Q8_0` and `Q4_0` matrices can also stay packed at 8 or 4 bits on
the device, where `x.matmul(w)` reads the levels directly:

```java
try (Gguf g = Gguf.open(Path.of("qwen2.5-0.5b-instruct-q8_0.gguf"))) {
    String arch = g.architecture();                       // "qwen2"
    long layers = (Long) g.metadata().get(arch + ".block_count");
    Tensor wq = g.packedTransposed("blk.0.attn_q.weight", Device.gpu());   // [in, out], 8-bit
}
```

### Pretrained tokenizers

`PretrainedTokenizer` reads a Hugging Face `tokenizer.json`, a SentencePiece `.model` or the
vocabulary inside a GGUF file, and produces the same ids as the library the model was
trained with (checked against Hugging Face tokenizers and sentencepiece). It covers
byte-level BPE (GPT-2, Llama 3, Qwen), SentencePiece BPE with byte fallback (Llama 2,
Mistral), Unigram (T5) and WordPiece (BERT).

```java
PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("Qwen2.5-0.5B-Instruct"));
int[] ids = tok.encode("<|im_start|>user\nHi<|im_end|>\n");   // special tokens in text are recognized
String text = tok.decode(ids, true);                           // skipping special tokens
int eos = tok.eos().orElseThrow();

String prompt = tok.applyChatTemplate(List.of(              // the model's own Jinja chat template
        ChatTemplate.message("system", "You are terse."),
        ChatTemplate.message("user", "Hi!")), true);
```

### Running a pretrained language model

`Llama` loads Llama-family models (Llama 2 and 3 with their rope scaling, Mistral, Qwen2,
SmolLM, TinyLlama) from a Hugging Face directory or a GGUF file. Q8_0 and Q4_0 GGUF
matrices stay packed in memory. Its logits, greedy and beam-search output are checked
against transformers. `Generator` writes text with it: temperature, top-k, top-p, min-p,
repetition penalty, stop tokens and stop strings, streaming that never splits a character,
batches of prompts (padded on the left) and beam search. Each new token costs one step,
because the model keeps a key/value cache.

DeepSeek-V2 (and V2-Lite), DeepSeek-V3 and Kimi K2 load the same way, from Hugging Face
folders (V3's 8-bit float weights with their block scales included) or GGUF files.
Multi-head latent attention caches one compressed latent and one rotary key per token
instead of keys and values per head, and decoding attends over that latent directly, with
fused kernels on the CPU, Vulkan and CUDA. Their experts follow DeepSeek's routing: shared
experts, group-limited choice, softmax (V2) or sigmoid with a selection bias (V3), and
YaRN for long context. Tiny random models check the logits against transformers and the
GGUF reading against llama.cpp.
The reusable parts live in core: the `LatentAttention` layer (trainable, saved with a
model, with a latent cache for decoding), `Functional.latentAttention` and
`Functional.absorbLatentUp`, and `Functional.chooseExperts` (group-limited, with a
selection bias) with `Functional.routedExperts` for any mixture of experts.

```java
Path dir = Path.of("Qwen2.5-0.5B-Instruct");
Llama model = Llama.load(dir, Device.best(), DType.FLOAT16);   // or Llama.load(Path.of("model.Q8_0.gguf"), ...)
PretrainedTokenizer tok = PretrainedTokenizer.load(dir);
Generator gen = new Generator(model, tok);

String prompt = tok.applyChatTemplate(List.of(ChatTemplate.message("user", "Name three rivers.")), true);
Generator.Options options = Generator.Options
        .sampled(200, Sampling.temperature(0.7f).withTopP(0.9f).withRepetitionPenalty(1.1f), 42)
        .withStopStrings("<|im_end|>");
gen.generate(prompt, options, (row, text) -> System.out.print(text));   // streams as it writes

List<Generator.Output> many = gen.generate(List.of("Once", "In a land"), Generator.Options.greedy(50), null);
int[] best = gen.beamSearch(tok.encode("The capital of France is"), 4, 10, 1.0, null);
```

To chat with a downloaded model: `./gradlew :brainiac-examples:run --args="chat
models/Qwen2.5-0.5B-Instruct --device vulkan"` (a model directory or a `.gguf` file).

The model tests run on the CPU by default; `./gradlew :brainiac-llm:test -PtestDevice=vulkan`
(or part of a device name such as `9070`) runs them on that device. The backend offers the
one device it picked, so a second GPU also needs the backend's own
choice, such as `-Dbrainiac.vulkan.device=1` in `JAVA_TOOL_OPTIONS`.
`-PtestModels=<folder>` also runs real models from Hugging Face found in that folder (model
directories and `.gguf` files): each must answer a question correctly, and where
`reference_models.py` (in the test resources) has written transformers' output next to a
model, its logits and greedy answer must match.

### Serving, constrained output, long context and evaluation

`OpenAiServer` answers the OpenAI API's `/v1/models`, `/v1/completions` and
`/v1/chat/completions`, with streaming as server-sent events, so existing clients work with a
local model. With a batch size above one it decodes concurrent requests together:
`Generator.batcher` prefills each new request and joins it to the running batch, and a
finished request leaves at once. A chat's `response_format` with a JSON schema constrains the
answer, as `Constraint` does for `Generator`.

```java
Set<Integer> stops = Generator.stopTokens(dir, tok);
try (OpenAiServer server = OpenAiServer.start(new InetSocketAddress("127.0.0.1", 8080), "qwen", gen, tok,
        stops, 8)) {
    // curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
}

Constraint person = Constraint.jsonSchema("""
        {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
         "required": ["name", "age"]}""");
String json = gen.generate(prompt, Generator.Options.greedy(80).withStopTokens(stops).withConstraint(person));

try (Generator.Batcher batcher = gen.batcher(8)) {
    CompletableFuture<Generator.Output> answer = batcher.submit(prompt, Generator.Options.greedy(100), null);
}
```

Longer contexts: `LlamaConfig` reads YaRN and dynamic rope scaling, `model.slidingWindow(n)`
attends to the last `n` positions, and `model.prefillChunk(n)` reads long prompts in pieces.
`model.saveGguf(file, tok, true)` writes a GGUF file (float16 matrices) that llama.cpp and
`Llama.load` read back. `Evaluation` measures a model: `perplexity`, `logLikelihood`,
`multipleChoice` and `needleInHaystack`.

## Speech: speech-to-text and text-to-speech

`brainiac-audio` reads WAV files and runs published speech models: OpenAI's Whisper
(`openai/whisper-tiny` up to `large-v3`) for speech-to-text, and four text-to-speech models:

| Model | Folder | What it adds |
|---|---|---|
| VITS | `facebook/mms-tts-eng` (and other MMS languages) | small, many languages |
| Kokoro | `hexgrad/Kokoro-82M` | 28 English voices, blends such as `"af_heart*0.7+bm_george*0.3"`, custom style packs (`addVoice`), a built-in English phonemizer |
| F5-TTS | `SWivid/F5-TTS` (`F5TTS_v1_Base`) with `charactr/vocos-mel-24khz` next to it | clones a voice from a few seconds of audio and its transcript |
| Orpheus | `canopylabs/orpheus-3b-0.1-ft` with `hubertsiuzdak/snac_24khz` next to it | eight voices and emotion tags in the text: `<laugh>`, `<sigh>`, `<gasp>`... |

Each matches its reference implementation on tiny random models.

```java
try (TextToSpeech tts = SpeechModels.textToSpeech(Path.of("mms-tts-eng"), Device.cpu(), DType.FLOAT32);
     SpeechToText stt = SpeechModels.speechToText(Path.of("whisper-base"), Device.cpu(), DType.FLOAT32)) {
    Audio speech = tts.synthesize("The weather is nice today.", SpeechOptions.defaults(), chunk -> play(chunk));
    Wav.write(speech, Path.of("weather.wav"));

    Transcription t = stt.transcribe(Wav.read(Path.of("talk.wav")),           // any length, any sample rate
            TranscriptionOptions.defaults().language(null),                    // detect the language
            segment -> System.out.printf("%.1f-%.1f %s%n", segment.start(), segment.end(), segment.text()));
}
```

Whisper decodes long audio 30 seconds at a time with timestamps, falls back to higher
temperatures when a window repeats itself, skips silence, and can translate to English
(`Task.TRANSLATE`). `SpeechModels.kind(folder)` tells which kind of model a folder holds
without loading it. Both take a cancellation check.

`TextToSpeech` says what a model can do: `voices()`, `canMixVoices()`, `canClone()` and
`tags()`. Cloning takes the reference in the options; F5-TTS also lists `name.wav` /
`name.txt` pairs in a `voices` folder next to its checkpoint as voices.

```java
try (TextToSpeech f5 = SpeechModels.textToSpeech(Path.of("models/f5-tts"), Device.cpu(), DType.FLOAT32);
     TextToSpeech orpheus = SpeechModels.textToSpeech(Path.of("models/orpheus-3b"), gpu, DType.BFLOAT16)) {
    Audio me = Wav.read(Path.of("me.wav"));
    Audio clone = f5.synthesize("This is not really me talking.",
            SpeechOptions.defaults().reference(me, "What I said in me.wav."), null);
    Audio fun = orpheus.synthesize("That is hilarious <laugh> really.", SpeechOptions.defaults().voice("leo"), null);
}
```

The building blocks are ordinary layers, so self-built models can use them too:
`MelSpectrogram` (raw audio to log-mel, including `MelSpectrogram.whisper(80)`),
`RelativeAttention` (windowed relative positions), `HifiGanBlock` (vocoder residual block),
`WaveNet` (gated dilated stack with optional conditioning),
`Flows.rationalQuadraticSpline` (neural spline flows), style-conditioned `StyleNorm`,
`AdainResBlock` and `AdainSnakeBlock` (StyleTTS2), `ConvNeXt1d` (with optional GRN and layer
scale), `DitBlock` (adaLN-Zero diffusion transformer block with rotary positions) and
`Snake`, with Slaney mel filters, `Signal.istftSame` and `Functional.rotaryInterleaved`.

```java
Model keywordSpotter = Model.sequential()
        .input(16_000)                          // one second of 16 kHz audio
        .melSpectrogram(16_000, 400, 160, 40)   // [40 mels, 101 frames]
        .conv1d(64, 3, Activation.GELU)
        .hifiGanBlock(3, 1, 3, 5)
        .globalAvgPool()
        .dense(12, Activation.SOFTMAX)
        .build();
```

## Image generation: Stable Diffusion and Flux

`brainiac-diffusion` runs pretrained Stable Diffusion 1.x, 2.x and XL models. It loads a
Hugging Face diffusers folder (`model_index.json` with `unet`, `vae`, `text_encoder` and
`tokenizer` folders) or a single `.safetensors` checkpoint as image tools share them:

```java
try (StableDiffusion sd = StableDiffusion.load(Path.of("models/stable-diffusion-xl-base-1.0"),
        Device.best(), DType.FLOAT16)) {
    ImageRequest request = ImageRequest.of("a red fox in snow, golden hour")
            .negativePrompt("blurry")
            .size(1152, 896)           // or sd.size(AspectRatio.LANDSCAPE_4_3)
            .steps(30)
            .seed(7)
            .build();
    List<GeneratedImage> images = sd.generate(request, Progress.withPreviews(5, (step, total, preview) -> {
        System.out.println(step + " / " + total);
        return true;                   // false stops the run (CancellationException)
    }));
    Files.write(Path.of("fox.png"), images.get(0).png());
}
```

- `StableDiffusion.detect(path)` says whether a folder or file is SD1, SD2 or SDXL without loading it.
- Samplers: Euler, Euler ancestral, DPM++ 2M, and the Karras-spaced variants (DPM++ 2M Karras is the default).
- Classifier-free guidance batches the prompt and negative prompt together; guidance 1 or less skips it.
- The UNet and text encoders run in the type you load them in (float16 on a GPU); the VAE runs
  in float32. Large convolutions are unrolled in bands of rows so a 1024 x 1024 decode stays
  within GPU memory.
- Previews are a cheap linear map of the latents, not a VAE decode.
- Single-file checkpoints carry no tokenizer: put `vocab.json` and `merges.txt` of
  `openai/clip-vit-large-patch14` in a `tokenizer` folder next to the file or in
  `~/.cache/brainiac/clip-tokenizer` (`StableDiffusion.downloadTokenizer` fetches them).

Each part was checked against diffusers and transformers on small random models: the CLIP
encoders, both UNet layouts, the VAE, the tokenizer, the noise schedules and whole runs of
the SD 1.x and SDXL pipelines (`brainiac-diffusion/src/test/python` makes the fixtures).

```sh
./gradlew :brainiac-examples:run --args="image models/stable-diffusion-v1-5 'a lighthouse at dusk' --steps 25"
```

### Flux

`Flux` runs Black Forest Labs' FLUX.1 dev and schnell from a diffusers folder: a multimodal
diffusion transformer sampled by flow matching, conditioned on T5-XXL and CLIP. In bfloat16 the
transformer takes about 24 GB and T5 about 9.5 GB; `Flux.Options.bits(8, 8)` stores both in
8 bits (4 bits quarters them), so it fits smaller GPUs. `Flux.Options.fitting(path, device,
dtype)` picks the most precise weights that leave room to generate on a device.

```java
try (Flux flux = Flux.load(Path.of("models/FLUX.1-schnell"), Device.best(), DType.BFLOAT16,
        Flux.Options.bits(8, 8))) {
    GeneratedImage image = flux.generate(ImageRequest.of("a red fox in snow, golden hour")
            .steps(4).build(), Progress.NONE).get(0);
}
```

- Schnell makes an image in 1 to 4 steps. Dev takes 20 to 50 steps and a guidance scale
  (about 3.5) that is an input to the model, so each step is one pass.
- Flux has no negative prompt and always steps with Euler.
- `ImageGenerator.load(path, device)` loads either family behind one interface. It picks the
  type (float32 on the CPU, bfloat16 for Flux and float16 for Stable Diffusion on a GPU) and
  quantizes Flux as far as the device's memory needs. A model that cannot fit fails before
  loading, with a message that says how much it needs. `detect(path)` names the family
  without loading, and `defaults()` gives each model's steps and guidance and whether it uses
  a negative prompt or a sampler choice.
- The transformer, T5, the tokenizer and whole dev and schnell runs were checked against
  diffusers and transformers on small random models (`flux_fixtures.py`).

### Wan video

`Wan` makes short videos with Alibaba's Wan 2.1 and 2.2 text-to-video models (Apache 2.0),
from a diffusers folder or from single files such as ComfyUI's
(`wan2.1_t2v_1.3B_fp16.safetensors`, `wan_2.1_vae.safetensors`, `umt5_xxl_fp16.safetensors`
plus umT5's `tokenizer.json`, all in one folder). The 1.3B model makes 832x480 clips on a
16 GB GPU. Wan 2.2 TI2V-5B, with its 16x autoencoder (48 latent channels, 2x2 patches), makes
1280x704 clips at 24 fps; the 14B models and Wan 2.2's two experts (high and low noise) want
a large GPU.

```java
try (Wan wan = Wan.load(Path.of("models/wan2.1-t2v-1.3b"), Device.best(), DType.BFLOAT16)) {
    GeneratedVideo video = wan.generate(VideoRequest.of("a red fox runs through fresh snow")
            .withFrames(33), Progress.NONE);
    Files.write(Path.of("fox.png"), video.apng());   // animated PNG; video.avi() for players
}
```

```
./gradlew :brainiac-examples:run --args="video models/wan2.1-t2v-1.3b 'a red fox runs through fresh snow' --frames 33"
```

Image-to-video starts from a picture, which is scaled to cover the video's size and cut to it:

```java
GeneratedVideo video = wan.generate(VideoRequest.of("the waves roll in").withSize(1280, 704),
        picture, Progress.NONE);   // wan.input() says TEXT, TEXT_OR_IMAGE or IMAGE
```

| Model | How the picture conditions | Files |
| --- | --- | --- |
| Wan 2.2 TI2V-5B | replaces the first latent frame, held at noise level 0 | the same as for text |
| Wan 2.2 I2V-A14B | a mask and the latents of the picture plus black frames as extra channels; two experts | high and low noise transformers |
| Wan 2.1 I2V 14B (480p, 720p) | the same channels, and cross-attention to the picture's CLIP ViT-H/14 features | plus `clip_vision_h.safetensors` |

```
./gradlew :brainiac-examples:run --args="video models/wan2.2-ti2v-5b 'the waves roll in' --image beach.jpg --size 1280x704 --frames 121"
```

On a rented machine with several GPUs, `Wan.plan` reads the files' headers and the devices'
free memory and places the parts for a video size: Wan 2.2's two experts on different GPUs,
umT5 and the CLIP encoder where there is room, each transformer's GPU keeping room for a
step's working memory and the VAE's GPU room to decode, as measured on an A100: Wan 2.1
14B at 832x480 and 9 frames held 44 GB of weights, 26 GB more while sampling and 18 GB to
decode; 1.3B at 832x480 and 33 frames 9 and 17 GB. It
falls back to 8- then 4-bit weights when nothing else fits. The example's `--device` takes a
comma-separated list.

```java
Wan.Plan plan = Wan.plan(files, Device.available(), DType.BFLOAT16, request);
System.out.print(plan.describe());
try (Wan wan = Wan.load(files, plan.placement(), DType.BFLOAT16, plan.options())) { ... }
```

| Model (bfloat16) | Weights | One GPU |
| --- | --- | --- |
| Wan 2.1 / 2.2 T2V and I2V 14B | 28.6 GB + umT5 11.4 GB (+ CLIP 1.3 GB) | 80 GB at 832x480, short clips |
| Wan 2.2 A14B (two experts) | 2 x 28.6 GB + umT5 11.4 GB | two 80 GB GPUs at 832x480 |
| Wan 2.2 TI2V-5B | 10 GB + umT5 | about 56 GB at 1280x704 and 121 frames |

Activations still take several times what diffusers needs, so 720p and five seconds with a
14B model is more than one 80 GB card holds for now; `Wan.plan` says so before loading.

- Frames are `1 + 4k` (81 is five seconds at 16 fps); width and height multiples of 16.
- The VAE decodes one latent frame at a time, so memory does not grow with the length.
- umT5, the transformer (with per-patch timesteps and image tokens for image-to-video), both
  autoencoders, UniPC and the whole pipeline were checked against diffusers on small random
  models (`wan_fixtures.py`, `wan22_fixtures.py`); the three image-to-video pipelines,
  CLIP vision and Pillow's bicubic and Lanczos resizing too (`wan_i2v_fixtures.py`).

### The same layers in your own networks

Stable Diffusion runs on ordinary brainiac-nn layers in `io.github.zzuegg.brainiac.nn.diffusion`.
They train, save and load like every other layer, so you can build your own latent diffusion
model from them, or fine-tune a pretrained one:

| Layer | What |
| --- | --- |
| `TimestepEmbedding` | sinusoids of a step (or any number) through two dense layers |
| `ResnetBlock` | group norm, SiLU, 3x3 convolutions, plus an optional time embedding |
| `SpatialTransformer` | transformer blocks over an image's pixels, cross-attending to a context (GEGLU feed-forward) |
| `ImageAttention` | single- or multi-head self-attention over pixels |
| `UNet` | the full denoising UNet; also a `Denoiser` for `Diffusion.loss` |
| `AutoencoderKL` | the VAE: `encode`, `decode`, `toLatents`, `loss` (reconstruction + KL) |
| `ClipTextEncoder` | CLIP's causal text transformer: last and penultimate layers, pooled embedding |
| `ClipVisionEncoder` | CLIP's vision transformer: patches, class token, pre-norm blocks |
| `T5Encoder` | T5's encoder: relative position biases, RMS norms, gated feed-forward |
| `FluxTransformer` | a multimodal diffusion transformer (MMDiT): joint text-image blocks, single-stream blocks, axial rotary positions, adaLN-Zero |
| `WanTransformer` | a video diffusion transformer: 3D patches and rotary positions, cross-attention to text (and image) tokens, adaLN |
| `WanVae` | a causal 3D VAE that encodes and decodes video frame chunk by chunk (Wan 2.1's 8x and Wan 2.2's 16x) |

```java
UNet unet = new UNet(UNet.Settings.of(4, 4, new int[] {64, 128, 256}, 512), new SplittableRandom(1));
Diffusion diffusion = new Diffusion(NoiseSchedule.scaledLinear(1000, 0.00085, 0.012), Diffusion.Prediction.NOISE);
Denoiser model = unet.conditional().given(textEmbeddings);   // [batch, tokens, 512]
Var loss = diffusion.loss(model, latents, random);

// sample in 25 steps with DPM-Solver++ instead of 1000 DDPM steps
double[] sigmas = SigmaSampler.karras(diffusion.schedule(), 25);
Tensor start = Tensor.normal(shape, 0, 1, random).mul((float) sigmas[0]);
Tensor images = SigmaSampler.sample(SigmaSampler.model(model, diffusion.schedule(), Diffusion.Prediction.NOISE),
        start, sigmas, SigmaSampler.Method.DPM_PP_2M, random, SigmaSampler.Listener.NONE);
```

Flow matching works the same way: `FlowSampler.loss(model, data, random)` trains a velocity model,
and `FlowSampler.sample(...)` walks from noise to data with Euler steps on a shifted schedule.

```java
FluxTransformer dit = new FluxTransformer(FluxTransformer.Settings.of(16, 8, 64, 4, 8, 512, 512, false), random);
Denoiser model = dit.denoiser(textTokens, pooledText, 0);
Var loss = FlowSampler.loss(model, latents, random);
Tensor out = FlowSampler.sample(FlowSampler.model(model), noise, FlowSampler.sigmas(28, 3),
        SigmaSampler.Listener.NONE);
```

Inference runs inside `Var.noGrad(...)`, which records no graph: intermediate values are freed
right away, and attention uses the fused kernel. `Tensor.withoutValues(...)` builds a model
without allocating initial weights, for loading pretrained ones into it.

## Reinforcement learning

`brainiac-rl` has environments to act in and agents that learn from rewards.

Environments implement `Environment`: `reset(random)` and `step(action)`, with
`Space.discrete(n)` or `Space.box(low, high)` actions.

| Environment | Actions | What |
| --- | --- | --- |
| `CartPole` | 2 | balance a pole, +1 per step, up to 500 |
| `MountainCar` | 3 | build momentum to climb a hill (sparse; optional `shaped(true)`) |
| `Pendulum` | continuous `[-2, 2]` | swing up and hold a pendulum |
| `GridWorld` | 4 | maps from text: walls, holes, cliffs, `slippery(p)`; `cliff()`, `lake()` |
| `Bandit` | k | multi-armed bandit |

`env.limit(steps)` adds a time limit. Write your own environment by implementing the interface.

```java
Space actions = new CartPole().actionSpace();

Tabular.qLearning(4).learningRate(0.5).epsilon(Schedule.constant(0.1));    // also sarsa(), EXPECTED_SARSA

Reinforce.of(Nets.mlp(4, 2, 1, 32), actions)               // policy gradient
        .baseline(Nets.mlp(4, 1, 2, 32)).entropyBonus(0.01f).episodesPerUpdate(4);

Ppo.of(policy, value, actions)                              // PPO: GAE, clipping, minibatch epochs
        .rolloutSteps(256).epochs(10).minibatchSize(64).clipRange(0.2f).gaeLambda(0.95f)
        .clipValue(true).targetKl(0.02f).annealLearningRate(true).entropyBonus(0.0f);
Ppo.a2c(policy, value, actions);                            // A2C settings
ppo.fit(CartPole::new, 8, 200_000, RlListener.every(50));   // 8 environments in step

Dqn.of(() -> Nets.mlp(4, 2, Activation.RELU, 1, 64, 64), 2)  // DQN
        .doubleQ(true).dueling(false).nSteps(3)
        .prioritized(0.6, Schedule.linear(0.4, 1, 50_000))
        .targetUpdateEvery(1_000)                           // or softTargetUpdate(0.005f)
        .epsilon(Schedule.linear(1, 0.05, 10_000)).bufferSize(50_000).batchSize(64);

RlResult result = agent.fit(new CartPole(), 50_000, RlListener.every(20));
Rl.evaluate(agent, new CartPole(), 10, seed);               // mean greedy return
Rl.watch(agent, new CartPole(), seed, 100);                 // text frames of one episode
```

Policies output logits for discrete actions, or Gaussian means for continuous ones (a
learned log standard deviation sets the spread). The networks are ordinary `Model`s, so
they can be saved, moved to the GPU, or built from any layers. `ReplayBuffer` and
`PrioritizedReplayBuffer` can also be used on their own.

`./gradlew :brainiac-examples:run --args=rl` (about 40 seconds) does three things:

- On the cliff walk, Q-learning takes the edge (-12) and SARSA the safer path (-16).
- On cart-pole, REINFORCE and A2C reach the maximum of 500, PPO about 490 and DQN about 440. A random agent gets 23.
- On the pendulum, PPO goes from -1786 to about -186, holding it within about 6 degrees of upright.

## A growing brain

`Brain` is a layer made of named regions of neurons joined by sparse projections. While it
learns, `Plasticity` changes its structure. Each region has a fixed capacity, of which some
neurons are active. Each projection is a weight matrix with a mask. Shapes never change, so
a brain trains with any optimizer, runs on the GPU, and saves like any other layer.

```java
Brain.Spec brain = Brain.builder()
        .region("sensory", 64).initialNeurons(16).activation(Activation.RELU)
        .region("association", 32).initialNeurons(8)
        .region("motor", 16)
        .connect("input", "sensory", 0.1)            // 10% of possible connections
        .connect("sensory", "association", 0.2)
        .connect("association", "motor", 0.2)
        .connect("association", "sensory", 0.1)      // feedback: reads the previous tick
        .connect("motor", "association", 0.2)
        .output("motor")
        .ticks(3)                                     // activity settles over 3 steps
        .build();
Model model = Model.sequential().input(256).add(brain).dense(3, Activation.SOFTMAX).build();

Plasticity plasticity = Plasticity.of(model)
        .rewire(25, 0.3).until(1_000)                // prune weakest 30%, regrow as many, cosine to 0
        .growth(Plasticity.Growth.GRADIENT)          // where the gradient is largest (RigL), or RANDOM (SET)
        .neurogenesis(4, 0.02, 8)                    // loss stalls 4 epochs: wake 8 neurons
        .pruneDeadNeurons(1e-4)                      // put silent neurons back to sleep
        .verbose(true);
Trainer.of(model).listener(plasticity).fit(data);
plasticity.events();   plasticity.history();   brain.describe();
```

You can also drive a `Brain` by hand with `prune(fraction)`, `grow(projection, count,
byGradient, random)`, `addNeurons(region, count, random)` and `removeDeadNeurons(threshold)`.
`regions()`, `projections()`, `connections()` and `growthPressure()` inspect it.
`TrainingListener.onBatch` is the per-step hook that `Plasticity` uses.

`./gradlew :brainiac-examples:run --args=brain` runs two demos:

- **Five spiral arms.** A brain that starts with 8 neurons grows to about 87 whenever its
  loss stalls, and reaches 98.6% test accuracy.
- **Bars and crosses.** A three-region brain with feedback keeps about 2,500 connections,
  13% of a dense network with the same neurons. It scores 77% against the dense network's
  85%, and against 64% for a dense network with the same number of connections.

## Evolving network designs

`brainiac-evolve` searches for a better design with a genetic algorithm. It starts from a
design known to work (the template), optionally with fewer neurons, and evolves its
**genome**: named regions of neurons, each neuron with its own id and activation, joined by
projections that list exactly which connections exist, plus the learning rate, optimizer
and ticks. A genome is built into a `Brain` layer between the task's input (or a fixed front
end such as convolutions) and a fixed output layer.

Every candidate is trained briefly and scored on a validation split of the training data;
the best few are trained again with a second seed so a lucky score cannot win. Elites are
kept, parents are picked by tournament, children come from NEAT-style crossover (parents
lined up by neuron id) and mutation. The same seed gives the same evolution, also with
several threads.

```java
Evolution evolution = Evolution.of(Tasks.spiral())
        .template(Genome.dense(2, Activation.RELU, 64, 64))  // the hand-built 64-64 MLP
        .neuronScale(0.75)                // start from 3/4 of its hidden neurons
        .mutations("connections,neurons,activations,learning-rate")
        .population(16).generations(8)
        .epochs(40)                       // training budget per candidate
        .sizePenalty(0.01 / 1000)         // optional: one percent per thousand parameters
        .threads(4);
Evolution.Result result = evolution.run();
System.out.println(result.best().genome().describe());
Benchmark.Report evolved = evolution.benchmark(result.best().genome(), "evolved");
```

Which genes change is exactly the set of mutations you enable, each with its own rates and
bounds (`Mutations.neurons(rate, min, max)`, `Mutations.activations(rate, Scope.NEURON,
choices...)`, ...), and `Mutation.named(...)` adds your own:

| Mutation | Changes |
| --- | --- |
| `connections` | flips single connections on or off |
| `density` | grows or thins a whole projection |
| `projections` | adds a feed-forward, feedback or lateral projection, or removes one |
| `neurons` | adds or removes neurons in a region, wiring new ones at the projection's density |
| `activations` | gives single neurons another activation (`Activation.perUnit`) |
| `region-activations` | gives a whole region another activation |
| `switching` | lets single neurons choose their activation by their input, or stops them (`Brain.Switching`) |
| `switch-mode`, `switch-candidates` | flips a region's switching between soft and hard, or adds or removes a candidate |
| `learning-rate`, `optimizer`, `ticks` | training and recurrence hyperparameters |
| `regions` | inserts a new region (a layer, in effect) or removes one |

Fitness is the task's metric on the validation data (accuracy, or minus the mean absolute
error), optionally minus a size or time penalty (`Fitness`); ties go to the smaller design.

Two switches change how candidates train:

- `batched(true)` (`--batched`) trains a generation as **one stacked model**: every weight
  gets a population axis, each projection is one batched matrix product for all
  candidates, and missing neurons and connections are masked. Each candidate still trains
  exactly as it would alone (same start, batches and element-wise optimizer, its own
  learning rate folded into its weights). On a GPU backend a spiral generation
  needed 2,265 kernel launches instead of 9,464, with identical scores; it is meant for
  GPUs, where launches dominate for small networks, and is **on by default when training
  on a GPU** (`--one-by-one` turns it off).
- `inheritWeights(true)` (`--inherit`) starts every child from its parents' trained
  weights wherever it shares a connection, neuron or output with them (matched by neuron
  id); only new parts start random.

Per-neuron activations are computed as `sum_k f_k(x) * mask_k`, element-wise only, so the
Vulkan backend fuses them: forward and backward each take one kernel launch, the same as a
single shared activation, even with all seven default activations mixed. On a GPU every
mixed region computes all seven, used or not, so candidates that differ only in their mix
of activations share one generated kernel instead of compiling new ones every generation.

A **switching** neuron is the evolvable form of `SwitchingDense`: besides its weighted sum
`z` it gets control weights on the same incoming connections, a control value
`c = x V + d`, and computes `sum_k softmax_k(c u_k + e_k) f_k(z)` over its region's
candidates (relu, tanh, sigmoid and identity to start), so what arrives decides which
activation it uses. Which neurons switch, soft or hard mode, and the candidate set are all
genes; control weights are inherited by neuron id and train stacked like the rest. On
sentiment (a bag-of-words template, population 12, 6 generations) adding the three switching
genes moved the evolved design from 69.3% to 71.5%, with 9 of its 52 neurons switching;
the hand-built MLP gets 78.7%, so that task needs a design that reads word order more than
it needs a new gene.

```
./gradlew :brainiac-examples:run --args="evolve spiral --threads 4"
./gradlew :brainiac-examples:run --args="evolve sentiment --mutate connections,neurons,activations,switching,switch-mode,switch-candidates"
./gradlew :brainiac-examples:run --args="evolve shapes --device gpu"
./gradlew :brainiac-examples:run --args="evolve iris --mutate connections,neurons,activations --scale 0.5"
```

Results with the defaults (population 16, 8 generations, start from 3/4 of the neurons) on
a 4-core CPU, both models then trained with the task's own settings and tested on the test data:

| Task | Hand-built | Evolved | Parameters in use | Search time |
| --- | --- | --- | --- | --- |
| xor | 8 tanh, 100% | 3 tanh neurons, 4 of 6 connections, 100% | 33 → 11 | 3 s |
| iris | 16 relu, 100% | 10 neurons, 96.7% (one flower) | 131 → 77 | 2 s |
| spiral | 64-64 relu, 99.0% | 45-44 neurons, mostly relu, 99.3% | 4,547 → 2,190 | 67 s |
| shapes (population 8, 4 generations) | CNN + 64 relu, 100% | same convs + 48 mixed neurons, 99.4% | 37,875 → 27,393 | 66 s |

With `--mutate connections,neurons,activations` (no learning-rate gene), xor shrinks to 2
neurons, one tanh and one silu.

## Brainiac Studio

`brainiac-studio` is a local web app for the whole library: chat with models, generate with
them, build models in a node editor whose sockets only accept matching shapes, train them,
and let agents do the same with the studio's tools. The Java process is the server (the JDK's own HTTP
server, no dependencies) and the page is plain HTML, CSS and JavaScript.

```sh
./gradlew :brainiac-studio:run                     # then open http://127.0.0.1:7860/
./gradlew :brainiac-studio:run --args="--port 0"   # any free port
./gradlew :brainiac-studio:run --args="--data D:/BrainiacData"   # keep everything there
```

Everything the studio keeps goes to one **data folder**: `--data <folder>`, or else
`$BRAINIAC_STUDIO_DATA`, or else `~/.brainiac/studio`. The layout menu shows which one is in use.

| in the data folder | what |
|---|---|
| `keys.properties` | API keys, readable only by your account |
| `providers.properties` | models shown, Auto and addresses per provider (no keys) |
| `chats/` | one JSON file per conversation |
| `page.json` | what the page remembers: open tabs and panes, layout, density, size, chosen models, folded groups |
| `usage.jsonl` | the usage log |
| `outputs/` | every picture and take |
| `graphs/`, `models/` | models drawn in the builder, and the `.brainiac` files built and trained from them |
| `runs/` | every training run: its settings, each epoch and the loss curve |
| `datasets/` | your data for training: CSV files, MNIST-style IDX files and image folders |
| `downloads/` | models downloaded from Hugging Face |
| `local-models.txt` | model files and folders added on the Models screen |
| `hosts.properties` | other machines to train on, with their tokens, readable only by your account |
| `agents/` | every agent run: its goal, each step, tool call and answer |

Nothing is kept in the browser, so every browser on the PC shows the same conversations and
tabs; what an older version kept in a browser moves to the data folder the first time that
browser opens the studio.

It listens on this machine only, answers only requests addressed to `127.0.0.1` or
`localhost`, and its API needs the token it creates at launch (the page it serves carries
it). `GET /api/capabilities` lists what the installed modules can do, found at startup; a
module adds its own by providing `io.github.zzuegg.brainiac.studio.spi.CapabilityProvider`,
and an available capability replaces a planned placeholder of the same key:

```java
public final class MyTts implements CapabilityProvider {
    public List<Capability> capabilities() {
        return List.of(Capability.available("speech.synthesize", Medium.SPEECH, "Text to speech",
                "Speak text with a chosen voice", "my-tts"));
    }
}
// module-info.java: provides io.github.zzuegg.brainiac.studio.spi.CapabilityProvider with MyTts;
```

**Chat** works with models on this PC and at Anthropic, OpenAI, Google, Hugging Face, DeepSeek,
Qwen (Alibaba Cloud), Kimi (Moonshot), GLM (Zhipu), MiniMax, Mistral, xAI, Groq and OpenRouter.
Local models are Hugging Face model folders (Llama, Qwen2, Mistral) and `.gguf` files in
`./models`, `~/.brainiac/models`, `$BRAINIAC_MODELS`, or the folders given with
`--models <folder>`. **Add folder or GGUF** on the Models screen adds any other file or folder,
picked by browsing this PC or typed; the files stay where they are, and the list is kept in
`local-models.txt` in the data folder. **Hugging Face** on the same screen searches the Hub
and shows only the models the studio runs: whole model folders of a family `LlamaConfig.MODEL_TYPES`
names, and GGUF files of an architecture in `LlamaConfig.GGUF_ARCHITECTURES`. Each model's files
say whether they run, and why not, such as a quantization `Gguf.formats()` does not read or a GGUF
split into parts. The studio keeps no list of its own, so a family or format the
core learns shows up there by itself. A GGUF file's architecture is also read from its first
megabyte before anything is downloaded.
Gated models (Llama, Gemma) need a Hugging Face read token and the model's license accepted on
its page. Paste the token in the Hugging Face panel: it is kept in `~/.brainiac/hf_token`, readable
only by your account, where the core's downloader finds it too. A key saved under the Hugging Face
provider comes first, then that file, then `HF_TOKEN`, then the file `huggingface-cli login` writes.
The token is sent only to huggingface.co, never to the download servers it redirects to.
Downloads go to `downloads/` in the data folder, resume where they stopped, are checked against
the Hub's SHA-256, and use your Hugging Face key, if you added one, for gated models; the key
goes only to huggingface.co, never to the file servers it sends downloads from. A finished
download offers **Chat with it**. Remote providers list their own models live; add API keys under Models,
Providers. Each provider can hold several named keys: they are tried in order, and the next one
takes over when a key is refused, rate limited or out of credit. Saved keys go to
`keys.properties` in the data folder, readable only by your account. A key in the provider's
environment variable counts as one more key:

| provider | environment variable | balance shown |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | no |
| OpenAI | `OPENAI_API_KEY` | no |
| Google | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | no |
| Hugging Face | `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` | no |
| DeepSeek | `DEEPSEEK_API_KEY` | yes |
| Qwen (Alibaba Cloud, international) | `DASHSCOPE_API_KEY` | no |
| Kimi (Moonshot) | `MOONSHOT_API_KEY` | yes |
| GLM (Zhipu BigModel) | `ZHIPUAI_API_KEY` or `BIGMODEL_API_KEY` | no |
| MiniMax | `MINIMAX_API_KEY` | no |
| Mistral | `MISTRAL_API_KEY` | no |
| xAI | `XAI_API_KEY` | no |
| Groq | `GROQ_API_KEY` | no |
| OpenRouter | `OPENROUTER_API_KEY` | yes (the key's limit) |

The page never sees a key, only its name and last four characters. Auto sends a conversation
to a model on this PC when there is one, then to the providers with keys in the order above.
A gateway or proxy in front of a provider is set with `-Dstudio.<provider>.url=`, such as
`-Dstudio.openai.url=https://gateway.example/v1/` or `-Dstudio.qwen.url=` for Alibaba Cloud's
China region.

**Conversations** open in tabs. On a wide screen, Split (or Ctrl+click on a conversation) puts
two to four side by side, each with its own model, and each answers on its own, so several
can be written at once. Finish moves a conversation out of the tabs to the Finished list,
where Reopen brings it back. Conversations are saved in `chats` in the data folder, one JSON
file each, so they show up in any browser on this PC; the newest 1000 are kept.

**Edit** on a provider chooses which of its models appear in the model pickers and in Auto
(all, including new ones, or only the ones you pick), whether Auto may use the provider at all,
another address (a gateway, a proxy, or another region such as Qwen's mainland China endpoint),
and for Hugging Face the organization whose credits pay (`X-HF-Bill-To`). Hugging Face models
show how many hosts serve them and the cheapest host's price; models no host serves are left
out. The choices are saved in `providers.properties`, which holds no keys.
A model hidden from the pickers can still be asked for by name.

**Usage** is logged for every call, by provider, key, model and day, in
`usage.jsonl`: tokens as the provider counts them, pictures made,
characters spoken, time taken and failures. Prompts are not kept, and keys appear by name only.
Models, Providers shows the totals, a chart of the last 30 days and, where the provider
reports one, what is left on each key. Anthropic, OpenAI and Google only report spend to
organization admin keys, so the studio shows tokens for them but no balance, and it does not
turn tokens into money.

**The model builder** offers every layer the installed modules have, core and experimental,
with every option, found from their public API rather than from a list: a public static
factory, constructor or constant that makes a `LayerSpec` or `Layer` becomes a block, and the
chainable methods on what it makes (`Conv.of(32, 3, 3).stride(2)`) become its options. The
studio checks each graph with the core's own shape rules, so a block that cannot take its input
is refused with the core's reason, and a graph that checks out builds exactly as drawn. A new
layer shows up by itself; factories that need something the builder cannot fill in, such as a
tensor of weights, are left out. Parameter names come from the class files, which the build
compiles with `-parameters`.

On the Builder screen you drag blocks in from the palette, or click one to add it after the
selected block, and drag from an output to an input to wire them. Every block shows its output
shape. Its settings and options are in the side panel, and an option is a checkbox. A change
that would make a working block fail is refused and shown with the core's reason, so the canvas
never holds a broken model. **Build and save** writes the model to `models/` and the drawing to
`graphs/`, and **Open** brings a saved drawing back.

**Training** trains a built model on built-in data (spirals, circles, blobs, XOR, a sine curve,
and small image sets of strokes and stripes, so nothing has to be downloaded) or on files in
`datasets`: a CSV table (the last column is the answer unless you pick another), MNIST-style IDX
files, or a folder of images with one subfolder per class. The optimizers, losses, learning rate
schedules and metrics are found in the installed modules the same way as layers, so Muon, Lion,
Adafactor and the experimental ones are all there with their settings. The screen checks that the
data fits the model before a run starts and says what to change if it does not. Runs train one at
a time on this machine, with the loss of every step drawn live; a learning rate sweep starts one
run per rate. Stopping a run keeps what it learned. Each run saves its model as
`models/<model>-run<N>.brainiac`, which can be trained further.

Runs can also train on **another machine**. Start the studio there as a worker:

```sh
brainiac-studio --worker                          # port 7870 on every network address
brainiac-studio --worker --listen 192.168.1.20 --port 7900 --data D:/BrainiacWorker
```

It prints its addresses and a token. On the Train screen, **Add a host** takes the address and
the token, checks that the worker answers, and shows its devices and what it is training. Pick
it under **Train on** and the run is sent there: the model and the data files go over once (by
their SHA-256, so a file the worker already has is not sent again), the progress and the loss
curve come back live, and the trained model is saved to `models/` here as for a local run. A
worker trains only on files it was sent, keeps them in `<data>/worker`, and trains one run at a
time. The token is kept in `hosts.properties` and never sent to the page. The connection is
plain HTTP with the token, so use workers on your own network, over a VPN or through an SSH
tunnel (`ssh -L 7870:127.0.0.1:7870 ...` with the worker on `--listen 127.0.0.1`).

A worker also **serves its chat models**: the ones in its model folders (`--models`, else the
usual ones) and in `<data>/worker/downloads` and `<data>/worker/models`. Every host added on the
Train screen shows up on the Models screen under **Compute**, next to this PC and Vast.ai (the
hosted APIs are under **Model APIs**), as a provider named after the host, right after this PC, and
its models answer in Chat and for agents through the worker's own engine, streamed back as they
are written. Setting `BRAINIAC_WORKER_TOKEN` (32 characters or more) gives the worker that token
instead of the one it makes and keeps in `worker.token`, for machines that are set up by a script.

**Agents** work towards a goal with a chat model and the studio's tools: listing and asking
models, searching Hugging Face and downloading from it, listing data sets, checking and starting
training runs. Any chat model can drive one, on this PC or at a provider: the tools are
described in the system prompt in the Hermes `<tool_call>` format that Qwen and many other
models are trained on. The tools are generated by `Tools.from` from the methods of
`StudioTools`, so a method added there is a tool for every agent. In **Ask first** mode an agent
waits on the Agents screen before a tool that acts (downloading, training, asking a model that
may cost money); Allow or Deny it, with a note if you like. In **Auto** it calls every tool
itself. A finished agent takes another message and goes on. Programs that bring their own agent
call the same tools over `/api/tools` with the studio's token.

**Vast.ai** rents GPU machines by the hour. Paste an API key on the Training screen (it is kept in
`keys.properties`, or comes from `VAST_API_KEY`, and is sent only to Vast.ai) to see your balance,
the machines you rent, and verified offers to rent, cheapest first, filtered by GPU, memory, count
and price. **Rent…** on an offer shows its price and asks you each time, with the hours after
which the studio destroys the machine (2 by default, at most 48). The price is looked up again
before renting, and the studio refuses if it went up. A rented machine gets this checkout (its
`HEAD` commit), builds it with `cloud/vast/setup.sh` and runs a worker there, reached through an
SSH tunnel and added as a host, which takes 10 to 20 minutes and is billed too. Renting needs the
`ssh` command and your SSH key added at cloud.vast.ai, and the studio to run from a Brainiac
checkout (or `BRAINIAC_REPO` naming one). The Vast.ai card shows each machine's stage and cost so
far; **Destroy** ends it (and any other instance of your account). Machines are also destroyed when
setting them up fails and when the studio closes, and the setup's output is kept in `vast/` in the
data folder. The studio rents at most 3 machines at once.

**Ctrl K** finds anything: a screen, a conversation, a chat model, a saved model in the
builder, a training run, or a command such as a new conversation, a new model or another layout.

**Images and Voice** draw and speak through OpenAI (GPT image and DALL-E models, `tts-1` and
`gpt-4o-mini-tts` with their voices) and Google (Imagen and Gemini image models, Gemini speech
with its 30 voices), with the same keys. The screens only show settings the chosen model has.
Every picture and take is also saved to `outputs` in the data folder. A video added to the
Voice screen plays in the browser under each take and is never uploaded.

**Images on this PC**: a Stable Diffusion model (1.x, 2.x or XL, as a diffusers folder or a single
`.safetensors` checkpoint) or a FLUX.1 dev or schnell diffusers folder in a model folder shows up as
an image model of This PC, and Auto picks it before any provider. Stable Diffusion also takes what
to leave out, the steps, the guidance, the seed and the sampler; FLUX.1 takes the steps and the seed
(and dev the guidance), and the settings it ignores are not shown. FLUX.1 runs in bfloat16 on a GPU
with its weights quantized as far as the GPU's memory needs; a model that cannot fit is refused,
saying how much memory it needs. The picture shows a rough preview as it is drawn, **Stop** ends it, and each
picture keeps its seed, so **Reuse seed** draws it again. A single checkpoint needs the CLIP
vocabulary, which is fetched once from Hugging Face (about 1.6 MB) into
`~/.cache/brainiac/clip-tokenizer`.

**Voice on this PC**: a Kokoro folder (`hexgrad/Kokoro-82M`, with its `.pth` checkpoint and `voices`
folder) or a VITS model (such as `facebook/mms-tts-eng`) in a model folder speaks on the Voice screen
as a model of This PC, saving WAV files, and Auto picks it. Kokoro lists its 28 English voices by
name, and **Blend with** mixes a second voice in at the share you pick (sent as
`"voice": "af_heart*0.70+bm_george*0.30"`). An Orpheus folder (with SNAC next to it) speaks with
its 8 voices and offers its expression tags, such as `<laugh>`, as buttons that put them in the
text. An F5-TTS folder (with the Vocos vocoder next to it) speaks in voices made from recordings:
under **New voice**, name it, choose or **Record** 3 to 12 seconds of speech and say what is said
(left empty, a Whisper model writes it down). Such voices are kept in `voices` in the data folder. A Whisper model (such as
`openai/whisper-base`, with its `tokenizer.json`) writes down what is said: switch the Voice screen
to **Transcribe**, then drop a recording (any audio or video the browser plays; it is turned into
16 kHz mono in the browser) or **Record** one with the microphone. The text appears piece by piece,
with times, in the spoken language or in English, and **To chat** puts it in the chat box.
**Hugging Face → Speech** on the Models screen finds and downloads Whisper and VITS models with
the files they need (Kokoro is not offered there yet: copy its folder in).

```
GET  /api/models            providers, their keys (names only) and models, and what Auto picks
POST /api/keys              {"provider":"anthropic","label":"work","key":"..."} ("remove" or "first": true)
POST /api/chat              {"model":"auto","messages":[{"role":"user","text":"Hi"}]} -> event stream:
                            route, thinking, text ..., done or error
POST /api/images            {"model":"auto","prompt":"...","aspect":"16:9","count":2}
POST /api/images/draw       {"model":"local/sdxl","prompt":"...","negativePrompt":"","aspect":"1:1","count":2,
                            "steps":30,"guidance":6,"seed":42,"sampler":"DPM_PP_2M_KARRAS"} -> event stream:
                            route, step and preview ..., image per picture, done or error
POST /api/images/options    {"model":"local/sdxl"} -> the samplers and the model's usual steps and guidance
POST /api/speech            {"model":"auto","text":"...","voice":"coral","speed":1,"instructions":"calm"}
POST /api/speech/voice      {"model":"local/f5-tts","name":"Me","audio":"<base64 WAV>","transcript":""}
                            keeps a voice made from a recording ("remove": true forgets it)
POST /api/transcribe        {"model":"auto","audio":"<base64 WAV>","language":"de","translate":false} -> event
                            stream: route, segment ..., done (text, language) or error
POST /api/transcribe/options {"model":"local/whisper-base"} -> its languages and whether it translates
POST /api/providers         {"provider":"huggingface","auto":true,"models":["openai/gpt-oss-120b"],
                            "url":"","options":{"billTo":"my-org"}} (models null shows all)
POST /api/local             {"path":"D:/Models/qwen2.5-7b.gguf"} adds a model file or folder ("remove": true forgets it)
POST /api/local/browse      {"path":"D:/Models"} -> its subfolders and models to pick from (no path: where to start)
POST /api/hub/search        {"query":"qwen2.5 instruct","gguf":true} -> text generation models on Hugging Face
POST /api/hub/files         {"repo":"Qwen/Qwen2.5-0.5B-Instruct-GGUF"} -> its GGUF files and model folder, each
                            with its size and whether the studio runs it, and why not
POST /api/hub/download      {"repo":"...","option":"qwen2.5-0.5b-instruct-q8_0.gguf"} ("folder" for the whole model)
GET  /api/hub/downloads     every download with its progress; POST /api/hub/cancel {"id":"1"} stops one,
                            POST /api/hub/delete {"name":"..."} deletes a downloaded model
GET  /api/vast              the Vast.ai key (last 4 characters), balance and rented machines
POST /api/vast/key          {"key":"..."} saves the Vast.ai key in keys.properties ("remove": true removes it)
POST /api/vast/offers       {"gpu":"4090","minGpuRam":24,"gpus":1,"maxPrice":0.5} -> machines to rent, cheapest first
GET  /api/vast/rentals      the machines rented from the studio: stage, cost so far, host, error
POST /api/vast/rent         {"offer":123,"pricePerHour":0.36,"hours":2} rents it, unless it now costs more
POST /api/vast/destroy      {"id":456} destroys an instance; POST /api/vast/cancel {"key":"r1"} stops a setup
POST /api/hub/token         {"token":"hf_..."} saves the Hugging Face token to ~/.brainiac/hf_token ("remove": true
                            deletes it); GET /api/hub/downloads says which token is used, by its last 4 characters
GET  /api/usage             calls and tokens per provider, key, model and day, and the latest calls
GET  /api/balances          what is left on each key, at providers that report it
GET  /api/chats             the saved conversations, newest first
POST /api/chats             {"id":"k3x9a","title":"...","status":"open","messages":[...]} saves one;
                            {"id":"k3x9a","delete":true} deletes it
GET  /api/state             what the page remembers; POST {"layout":"ultra"} changes it (null removes)
GET  /api/builder/layers    every layer the installed modules offer, with its options, found from their API
POST /api/builder/check     {"graph":{"nodes":[...]}} -> each block's output shape, or why it cannot take its input
POST /api/builder/build     {"name":"resnet-s","graph":{...},"seed":42} -> models/resnet-s.brainiac, parameters, summary
GET  /api/builder/graphs    the saved graphs; POST {"name":"...","graph":{...}} saves one ("delete": true deletes)
GET  /api/train/options     every optimizer, loss, schedule and metric the installed modules offer, and the devices
GET  /api/train/data        the built-in data sets and what is in the datasets folder
GET  /api/train/models      the models that can be trained, with their input and output shapes
POST /api/train/check       {"spec":{...}} -> whether the model and the data fit, samples and steps per epoch
POST /api/train/start       {"spec":{"model":"resnet-s","data":{"kind":"builtin","name":"stripes"},
                              "optimizer":{"entry":"...Optimizer#adamW(float,float)","args":[0.001,0.01]},
                              "loss":{"entry":"...Loss#CROSS_ENTROPY"},"epochs":20,"batchSize":32,"validation":0.1}}
POST /api/train/stop        {"id":"3"} stops a run after its current step and saves what it learned
POST /api/train/delete      {"id":"3"} removes an ended run from the list (its model stays)
GET  /api/train/runs        every run with its progress, epochs and loss curve
GET  /api/hosts             the other machines to train on: online, devices, what each is training
POST /api/hosts             {"name":"Lab box","url":"192.168.1.20:7870","token":"..."} adds one after
                            checking it answers; {"id":"h1","remove":true} forgets one
GET  /api/tools             every tool with its JSON schema, and what it does if it acts
POST /api/tools/call        {"name":"search_hub","arguments":{"query":"qwen 0.5b"}} -> {"result":"..."}
GET  /api/agents            every agent run, newest first, and the tools
POST /api/agents/start      {"goal":"...","model":"auto","mode":"ask","maxSteps":12,"tools":["search_hub"]}
                            (mode "auto" calls tools that act without asking; tools left out: all)
POST /api/agents/get        {"id":"1"} -> the run with every step and message
POST /api/agents/answer     {"id":"1","allow":true,"note":"the small one"} answers the call it waits on
POST /api/agents/reply      {"id":"1","text":"..."} lets a finished run go on
POST /api/agents/stop       {"id":"1"}; POST /api/agents/delete {"id":"1"} removes an ended run
```

A worker answers only requests that carry its token (`Authorization: Bearer <token>`):

```
GET  /worker/info           name, devices, cores, memory, and what it is training and has queued
GET  /worker/models         the chat models it serves
POST /worker/chat           {"model":"...","system":"...","messages":[{"role":"user","text":"..."}],"maxTokens":1024}
                            answers as server-sent events: text, thinking, then done (or error)
GET  /worker/files/<sha256> whether it has a file; PUT with ?name= sends one (checked against the hash)
POST /worker/runs           {"spec":{...},"model":"sha:<sha256>/net.brainiac"} starts a run on files it was sent
GET  /worker/runs?id=1      the run's progress, epochs and loss curve
POST /worker/stop           {"id":"1"}
GET  /worker/result?id=1    the trained model
```

The other surfaces are being connected one at a time; until a surface reads real data it
shows example data and says so in the top bar.

## Benchmarks: test any model

`Benchmark` trains and tests any model on a fixed task, with the same data, split and
training settings every time, and reports test accuracy (or mean absolute error), loss,
per-class accuracy, the confusion matrix, the most common mistakes, parameter count,
training time and inference speed. `Tasks` in the examples has every example's problem
ready to use:

| task | input | output | what |
|---|---|---|---|
| `mnist` | `[1, 28, 28]` | `[10]` | handwritten digits (downloads 11 MB) |
| `fashion` | `[1, 28, 28]` | `[10]` | Fashion-MNIST clothing photos (downloads 30 MB) |
| `shapes` | `[1, 16, 16]` | `[3]` | synthetic bars and crosses |
| `waves` | `[24, 1]` | `[1]` | forecast the next value of a noisy wave |
| `reversal` | `[8, 6]` | `[8, 6]` | reverse a sequence of digits |
| `sentiment` | `[12]` | `[2]` | made-up reviews as word ids, with "not" and "but" |
| `spiral` | `[2]` | `[3]` | three interleaved spiral arms |
| `iris` | `[4]` | `[3]` | iris species from flower measurements |
| `xor` | `[2]` | `[1]` | exclusive or |

```java
Benchmark.Task task = Tasks.mnist();
Model mine = Model.sequential().input(1, 28, 28) /* ... */ .build();

Benchmark.Report report = Benchmark.run(task, "mine", mine);           // task's default training
Benchmark.Report quick = Benchmark.run(task, "mine", other, t -> t.epochs(1).batchSize(128));
Benchmark.Report saved = Benchmark.evaluate(task, "saved", Model.load(Path.of("mnist.brainiac")));
System.out.println(report);
System.out.println(report.confusionTable());

List<Benchmark.Report> all = Benchmark.compare(task,
        Map.of("small", () -> small(), "big", () -> big()), t -> t.epochs(2));
System.out.println(Benchmark.table(all));
```

A model fits a task when its input and output hold as many values as the task's shapes, so
a flat `input(784)` model works on MNIST too. Your own data becomes a task with
`Benchmark.Task.classification(name, train, test, inputShape, outputShape)` or
`regression(...)`.

From the command line, with built-in models (`default`, `linear`, `mlp`, `cnn`, `lenet`,
`bilstm`, `gru`, `transformer`) or saved model files:

```
./gradlew :brainiac-examples:run --args="bench mnist linear mlp lenet cnn --epochs 1"
./gradlew :brainiac-examples:run --args="bench offline"          # every task without a download
./gradlew :brainiac-examples:run --args="bench fashion my-model.brainiac --confusion"
```

## GPU and number formats

Add `brainiac-backend-vulkan` to run on a GPU. It runs Vulkan compute shaders called through
LWJGL (its only dependency, with LWJGL's shaderc bindings; the natives jars for the build
platform are picked automatically). It needs a Vulkan 1.2 driver with buffer device addresses and 64-bit shader
integers, which current AMD, NVIDIA and Intel drivers have. Allow LWJGL to call native code
with `--enable-native-access=org.lwjgl`.

```java
Device gpu = Device.best();          // the GPU if Vulkan finds one, else the CPU
Model model = ... .build().to(gpu);  // parameters move to device memory
Trainer.of(model).fit(data);         // batches follow the model
```

Every operation is one precompiled compute shader (GLSL in `src/main/glsl`, SPIR-V in the
resources; `./gradlew :brainiac-backend-vulkan:compileShaders` rebuilds them with
`glslangValidator`). Shaders reach tensors through device addresses passed as push
constants, so there are no descriptor sets to manage. Commands are recorded into command
buffers and submitted in batches, with a memory barrier only before a command that touches
memory an earlier one wrote, so the host waits for the device only when it reads values
back. Tensors are slots in large device memory blocks, pooled by size. Every shader is
compiled once per number format: float32 always; float64 (computed in double precision, with
the shaders' own exp and log, since GLSL has none for doubles) where the device has
`shaderFloat64`; float16 and bfloat16 (stored in 16 bits, computed in float32) where it has
16-bit storage buffers (AMD's LLPC compiler crashes on one float64 shader switching over every
function, so float64 unary operations and activations have one shader each).

Element-wise operations are fused: a result is recorded
instead of computed, later element-wise operations build on it, and when something else needs
the values (a matmul, a reduction, a download) the whole chain becomes one GLSL shader,
compiled at run time with shaderc. Transposes, slices and broadcasts are part of the chain and
are read in place through their strides. Pointers, shapes and constants are read from a small
table, so one shader serves every batch size and learning rate, and compiled shaders are kept
in `~/.cache/brainiac/vulkan` between runs. `brainiac.vulkan.fusion=false` runs every step on
its own; `brainiac.vulkan.cache` moves the cache (`none` for no cache).

System properties: `brainiac.vulkan.devices`
(`gpu` by default, `all` to include CPU drivers such as lavapipe), `brainiac.vulkan.device`
(index; a dedicated graphics card is preferred), `brainiac.vulkan.memory` (MB budget), `brainiac.vulkan.validate` (`true` turns on the Khronos
validation layer from the Vulkan SDK; `-Pbrainiac.vulkan.validate=true` for the tests),
`brainiac.vulkan.statistics` (`true` prints the counts below at exit), `brainiac.vulkan.profile` (`true` runs
every dispatch alone between GPU timestamps and adds the device time per shader to the statistics; the
generated shaders are written to `brainiac-fused` in the temp folder; `shapes` keeps times per work-group grid).
`VulkanBackend.status()` lists the devices found and `statistics()` counts dispatches,
barriers and submits. The examples take `--device vulkan` (or `gpu`) to run there. CI runs the tests on lavapipe.
Without a Vulkan driver the backend reports itself unavailable and its tests are skipped.

On an RX 9070 XT: the MLP trains 2 epochs of MNIST in 1.7 s; a 4096 x 4096 matmul reaches
6.5 TFLOP/s in float32 and up to 25 TFLOP/s in float16 on cooperative matrices; a
transformer on Shakespeare trains 3 epochs in 10.7 s; Qwen2.5-0.5B loads in 0.7 s and generates about 165 tokens/s
in float16. (An OpenCL backend existed until it was retired in favour of this one, which
was faster on every measure.)

`brainiac-backend-cuda` is a second GPU backend, for NVIDIA cards only. It calls NVIDIA's
libraries directly through Java's Foreign Function and Memory API, with no JNI and no
bindings library: the CUDA driver API (`nvcuda.dll`, `libcuda.so.1`, part of the display
driver), NVRTC to compile its kernels (`kernels.cu`) for the card at first use (cached in
`~/.cache/brainiac/cuda`), and cuBLAS for matrix products when it is installed (otherwise a
kernel of its own). NVRTC and cuBLAS come with the CUDA toolkit 11.2 or later, or NVIDIA's
`cuda` runtime container images; they are looked for on the `PATH`/`LD_LIBRARY_PATH`, in
`CUDA_PATH`/`CUDA_HOME` and in the usual install folders. Run with
`--enable-native-access=io.github.zzuegg.brainiac.backend.cuda` (or `ALL-UNNAMED`). With both
backends on an NVIDIA card, `Device.gpu()` and `Device.best()` take Vulkan unless
`-Dbrainiac.gpu=cuda`; `-PtestDevice=cuda` runs the model tests there. It covers every required
operation plus batched products, convolution windows, gather and scatter, RMSNorm, fused
attention (forward, backward and with a bias, heads up to 256, long key ranges split across
the device and combined), packed weights kept packed on the card (products of up to 16 rows read
the levels directly), float16/bfloat16 products of a few rows in a kernel of its own, and
Fourier transforms of power-of-two lengths up to 2048. Each GPU of the machine is a device
(`cuda`, `cuda:1`, ...); `DataParallel` over several of them sums the gradients with
`Tensor.sumAcross`, which uses NCCL's all-reduce (`libnccl.so.2`, Linux) between CUDA GPUs and
the host otherwise. System properties:
`brainiac.cuda.device` (index), `brainiac.cuda.memory` (MB budget), `brainiac.cuda.debug`
(`true` waits after every kernel, copy and product so an error names the operation that
caused it, and compiles with line information for `compute-sanitizer`), `brainiac.cuda.critical`
(`true` links the per-operation driver and cuBLAS calls as critical downcalls, which cost
less per call), `brainiac.cuda.tf32` (`true` lets float32 products use TensorFloat-32),
`brainiac.cuda.driver`/`nvrtc`/`cublas` (library files; `cublas=none` turns cuBLAS off),
`brainiac.cuda.cache`, `brainiac.cuda.statistics`, `brainiac.cuda.devices` (`one`: only the first
GPU), `brainiac.cuda.nccl` (library file, `none` to sum on the host), `brainiac.cuda.fusedAttention`
and `brainiac.cuda.fewRows` (`false` turns those kernels off). `CudaBackend.status()` names the
device and the libraries found, or why there is none; without an NVIDIA driver the backend
reports itself unavailable and its device tests are skipped (`-Pbench` prints product speeds).

`brainiac-backend-hip` is the same backend ported to AMD's HIP, which mirrors the CUDA API: the
HIP runtime (`amdhip64_7.dll`, part of the display driver and the HIP SDK; `libamdhip64.so` on
Linux), hipRTC to compile its kernels (`kernels.hip`, the CUDA kernels with AMD's half
conversions and shuffles) for the card at first use (cached in `~/.cache/brainiac/hip`), and
rocBLAS for matrix products when it has kernels for the card (otherwise a kernel of its own).
On Windows install AMD's HIP SDK (its core, libraries and runtime compiler) and put its `bin`
folder on the `PATH`; the libraries are also looked for in `HIP_PATH`, `ROCM_PATH` and the usual
install folders. The kernels are written for 32-lane wavefronts, so it needs an RDNA card
(Radeon RX 5000 or later; the RX 9070 XT is `gfx1201`). Run with
`--enable-native-access=io.github.zzuegg.brainiac.backend.hip`; the device is `hip`
(`-Dbrainiac.gpu=hip` to prefer it over Vulkan, `-PtestDevice=hip` for the model tests). It covers
the same operations as the CUDA backend; several GPUs sum gradients on the host (no RCCL on
Windows). System properties: `brainiac.hip.device`, `brainiac.hip.memory`, `brainiac.hip.debug`,
`brainiac.hip.critical`, `brainiac.hip.runtime`/`hiprtc`/`rocblas` (library files; `rocblas=none`
turns rocBLAS off), `brainiac.hip.arch` (the compiler target, when the runtime does not tell),
`brainiac.hip.cache`, `brainiac.hip.statistics`, `brainiac.hip.devices`,
`brainiac.hip.fusedAttention` and `brainiac.hip.fewRows`. `HipBackend.status()` names the device
and the libraries found, or why there is none.

Tensors and models come in four number formats:

```java
Model precise = model.to(DType.FLOAT64);              // double precision
Model small = model.to(DType.BFLOAT16);               // half the memory
Trainer.of(model).mixedPrecision(DType.FLOAT16);      // compute in float16, float32 weights,
                                                      // dynamic loss scaling
```

Mixed types promote to the wider one, gradients keep their value's format, and saved
models keep each tensor's format. On the CPU, float32 uses the fast kernels and the other
formats are computed in double precision; on the GPU, float64 needs `shaderFloat64` and
float16/bfloat16 are stored in 16 bits and computed in float32.

```
./gradlew :brainiac-examples:run --args="gpu"                     # devices and speed
./gradlew :brainiac-examples:run --args="bench mnist cnn --device gpu --mixed f16"
```

## Quantization

Weights can be stored in fewer bits: symmetric linear quantization rounds every weight
matrix and filter to a whole number from -127..127 (8 bits), -7..7 (4 bits) or any width
from 2 to 8 bits, times a float scale. One scale per output channel by default, per group
of inputs for low bit widths, or one per tensor:

```java
model.quantize(Quantization.int8());               // after training
model.quantize(Quantization.int4().groups(32));    // a scale for every 32 inputs of a unit
model.save(path);                                  // weights stored packed: ~4x / ~6x smaller
Model.load(path).quantization();                   // int4 (groups of 32)

Trainer.of(model)                                  // quantization-aware training: forward and
        .quantizationAware(Quantization.bits(2))   // backward with rounded weights, updates to
        .epochs(2)                                 // float weights (straight-through estimator)
        .fit(data);                                // ends quantized
```

Quantized are dense, convolution, attention, recurrent and embedding weights; biases,
normalization and activation parameters stay in floating point. A quantized model keeps
computing in its own number format with the rounded values, so it runs unchanged on the
CPU and the GPU. To also save memory and time, pack it: dense-like weights (dense,
attention, recurrent) then stay in memory as their 8- or 4-bit levels plus scales
(`Tensor.packed`), and matrix products read the levels directly, on the CPU (SIMD) and
in Vulkan. That pays off when generating text one token at a time, which waits
on reading the weights:

```java
model.quantize(Quantization.int4().groups(32)).pack();   // or Model.load(path).pack()
model.bytes();                                           // ~7x less weight memory
```

Generating with 6 transformer blocks of 768 (a GPT-2-small-sized model), tokens/s:

| device | float32 | int8 packed | int4 packed (groups of 32) |
| --- | --- | --- | --- |
| CPU, 32 threads, Vector API | 124 | 307 | 282 |
| RX 9070 XT, Vulkan | 355 | 802 | 888 |

One row times 4096x4096 weights takes 0.24 ms in float32 on that GPU and 25 µs in int8
(20 µs in int4); on the CPU 1.2 ms in float32 and about 0.12 ms packed.

Products of more rows (over 8 on a GPU, 16 on the CPU: training, batch inference) unpack the weights first, and
any other operation on a packed weight sees its values; `unpack()` turns them back into
float32 tensors. Models too small to wait on their weights, such as the Shakespeare
example, gain memory but not speed: there the cost per token is in launching work.

Files with quantized weights are format version 5; others are still written as version 4.
Compare schemes on any benchmark task:

```
./gradlew :brainiac-examples:run --args="bench mnist cnn --quantize int8,int4,int4/g32,int2 --qat 1"
./gradlew :brainiac-examples:run --args="bench shakespeare --quantize int8,int4/g32 --pack"   # + generation speed
```

## Experiments

`brainiac-experiments` is a lab for research ideas: new layers, optimizers, learning
rules, memory and context tricks, decoding methods, RL algorithms, generative toys and
more. Nothing in it is guaranteed to stay.

It is split in two:

- **`io.github.zzuegg.brainiac.experimental.*`** holds reusable building blocks
  (`experimental.nn`, `experimental.train`, `experimental.rl`, `experimental.llm`,
  `experimental.data`, `experimental.evolve`, `experimental.tensor`,
  `experimental.autograd`, `experimental.nn.diffusion`). Each package mirrors the core
  package of the same name. The blocks implement the core interfaces (`Layer` and
  `LayerSpec`, `Optimizer`, `Loss`, `Environment`, `Agent`, `Tokenizer`,
  `LanguageModel`, `TrainingListener`), so they drop into existing code. When a block
  proves itself, it moves to core by dropping `.experimental` from its package.
- **`io.github.zzuegg.brainiac.experiments.<theme>`** holds the experiments: small
  synthetic tasks that compare a technique against a baseline. Themes include agents,
  attention, context, control, evolution, exotic, generative, graphs, interp, language,
  learning, memory, optim, reasoning, selfsupervised, sequence, techniques, timeseries
  and vision.

Trying an experimental block is a one-line change:

```java
Model model = Model.sequential()
        .input(64)
        .dense(128, Activation.GELU)
        .add(DynamicTanh.spec(0.5f))          // experimental.nn: normalization-free LayerNorm
        .dense(10, Activation.SOFTMAX)
        .build();

Trainer.of(model)
        .loss(Loss.CROSS_ENTROPY)
        .optimizer(Sophia.of(1e-3f))          // experimental.train: second-order optimizer
        .fit(data);
```

Each experiment is a JUnit test that prints what it measured. Run them all with
`./gradlew :brainiac-experiments:test`. The measured results, grouped by theme, are in
[brainiac-experiments/RESULTS.md](brainiac-experiments/RESULTS.md). They come from
CPU-sized toy tasks, so treat them as directional, not as benchmarks.

## Performance

The CPU backend splits large operations across all cores (limit with
`-Dbrainiac.cpu.threads=N`); results are bit-identical for any thread count.

Matrix multiplication can also use SIMD instructions through Java's incubating Vector
API, which is about 8x faster. It is opt-in, because the JVM prints a warning for
incubator modules: start your program with `--add-modules jdk.incubator.vector`
(the examples and benchmarks already do). `CpuBackend.simd()` tells whether it is on.

| 4-core machine | plain Java | with SIMD |
| --- | --- | --- |
| matmul 1024x1024 | 15 GFLOP/s | 147 GFLOP/s |
| train MLP 784-256-128-10 | 8.2 k samples/s | 20 k samples/s |

Built-in activations run as fused kernels: one pass forward and one pass backward,
instead of a chain of operations that each read and write the whole tensor. With SIMD
on, element-wise arithmetic and the arithmetic-only activations (ReLU, leaky ReLU,
ReLU6, hard tanh/sigmoid/swish, softsign) use the Vector API too, with bit-identical
results. Your own activations can use them through `Var.activation(ActivationOp, param)`.

| 4-core machine, forward + backward, 4M values | composed | fused |
| --- | --- | --- |
| GELU | ~25 M/s | ~55 M/s |
| SiLU | ~50 M/s | ~90 M/s |
| leaky ReLU | ~100 M/s | ~600 M/s |
| hard swish | ~110 M/s | ~720 M/s |

Measure on your machine with:

```sh
./gradlew :brainiac-benchmarks:run                        # all cases
./gradlew :brainiac-benchmarks:run --args="--quick"       # shorter
./gradlew :brainiac-benchmarks:run --args="--threads 1"   # single-threaded
./gradlew :brainiac-benchmarks:run --args="--no-simd"     # without the Vector API
```
