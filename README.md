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
| `brainiac-backend-opencl` | Optional GPU backend: OpenCL through the Foreign Function and Memory API |
| `brainiac-backend-vulkan` | Optional GPU backend: Vulkan compute shaders through LWJGL (its only dependency) |
| `brainiac-autograd` | Reverse-mode automatic differentiation |
| `brainiac-nn` | Layers, activations, losses, initializers, models, the growing `Brain` |
| `brainiac-train` | Optimizers, training loop, metrics, listeners, `Plasticity` |
| `brainiac-data` | Datasets, batching, splitting, CSV loading, preprocessing |
| `brainiac-io` | File formats: JSON, memory-mapped safetensors (single or sharded) and GGUF weights, weight-name mapping, pretrained tokenizers |
| `brainiac-llm` | Pretrained language models (Llama, Llama 3, Qwen2 from Hugging Face or GGUF files), a KV cache, sampling, streaming, batched generation and beam search |
| `brainiac-rl` | Reinforcement learning: environments, replay buffers, Q-learning, SARSA, REINFORCE, A2C, PPO, DQN |
| `brainiac-evolve` | Neuroevolution: a genetic algorithm that searches wiring, neuron counts, per-neuron activations and hyperparameters |
| `brainiac` | Facade: depend on this one module to get the whole API |
| `brainiac-examples` | Runnable examples |
| `brainiac-benchmarks` | Benchmark harness for kernels and training speed |

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

The model tests run on the CPU by default; `./gradlew :brainiac-llm:test -PtestDevice=vulkan`
(or `opencl`, or part of a device name such as `9070`) runs them on that device.

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
  learning rate folded into its weights). On the CPU OpenCL driver a spiral generation
  needs 2,265 kernel launches instead of 9,464, with identical scores; it is meant for
  GPUs, where launches dominate for small networks, and is **on by default when training
  on a GPU** (`--one-by-one` turns it off).
- `inheritWeights(true)` (`--inherit`) starts every child from its parents' trained
  weights wherever it shares a connection, neuron or output with them (matched by neuron
  id); only new parts start random.

Per-neuron activations are computed as `sum_k f_k(x) * mask_k`, element-wise only, so the
OpenCL backend fuses them: forward and backward each take one kernel launch, the same as a
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

Add `brainiac-backend-opencl` to run on a GPU. It calls the graphics driver's OpenCL
library directly through Java's Foreign Function and Memory API, so there is nothing
native to build. Allow it to call native code with
`--enable-native-access=io.github.zzuegg.brainiac.backend.opencl` (or `ALL-UNNAMED` on
the class path).

```java
Device gpu = Device.best();          // the GPU if OpenCL finds one, else the CPU
Model model = ... .build().to(gpu);  // parameters move to device memory
Trainer.of(model).fit(data);         // batches follow the model
```

The same kernels run every model: layers are written once against tensor operations,
and each operation launches one precompiled OpenCL kernel. Memory of unreachable tensors
is returned when they are garbage collected. System properties:
`brainiac.opencl.devices` (`gpu` by default, `cpu` or `all` to include OpenCL CPU drivers
such as PoCL), `brainiac.opencl.device` (index; by default a dedicated graphics card is preferred over
integrated graphics), `brainiac.opencl.memory` (MB budget), `brainiac.opencl.profile` (`true` times every
kernel on the device and prints the totals at exit; also `OpenClBackend.profile()`).
`OpenClBackend.status()` lists the devices found. Buffers of collected tensors are pooled and reused, and shape tables
are cached on the device, so training does not create device memory on every operation.
Element-wise operations are fused: instead of one kernel per add, multiply or activation,
the backend records chains of them and generates one OpenCL C kernel per chain when the
values are needed (by a matmul, a reduction or a download); a reduction also computes its
input chain on the fly (long rows are split across a work-group), and reductions of the same
shape that are needed at the same time (such as a layer's gradient sums) run as one kernel. A reduction that reads
another of the same rows (a variance reading its mean, a softmax sum reading its maximum) runs as a second
pass of that kernel. A chain on a matrix
product (bias, activation, optimizer arithmetic) becomes the matmul kernel's epilogue.
Convolution and pooling windows are never unrolled into memory: the matmul computes each window
value as it loads it (forward and for the weight gradient), and a pooling reduction as it reduces;
the backward pass sums windows back into the image (col2im) and finds pooling maxima (argmax) inside
generated kernels too, together with the chains around them.
An optimizer step runs as one kernel for all parameters, writing the new values and the
optimizer state together (`Tensor.evaluate` asks for this from any code).
Filled tensors and scalar operands are kernel arguments, so a value that changes every step
neither needs a buffer nor a new kernel, and a result gets a device buffer only when it is
actually stored (most values in a chain never are).
Transposes, slices and
other strided copies are not copied either: they are read in place by the chain or matmul
that uses them. This works for any model, without per-layer kernels, and the
generated code is plain OpenCL C 1.2, so it runs on every vendor's driver. Shapes and
strides are kernel arguments, so a training loop needs a few dozen generated kernels.
They compile in the background (the operations run one by one until then) and are cached
on disk, so later runs start at full speed. Results are the same with and without fusion.
Properties: `brainiac.opencl.fusion` (`false` turns it off), `brainiac.opencl.fusion.async`
(`false` compiles before running), `brainiac.opencl.cache` (directory, default
`~/.cache/brainiac/opencl`, or `none`), `brainiac.opencl.flushEvery` (launches between
explicit queue flushes; some drivers otherwise hold commands back and leave the GPU idle.
Default 4; `-1` adapts: every launch right after the host waited, then doubling up to 64), `brainiac.opencl.fusion.store` (`true` also stores intermediates so chains
can span other kernels: fewer launches, but slower on some drivers).
Without an OpenCL GPU the backend reports itself unavailable and its tests are skipped;
CI runs them on PoCL.

### Vulkan

`brainiac-backend-vulkan` is a second GPU backend, on Vulkan compute shaders called through
LWJGL (its only dependency, with LWJGL's shaderc bindings; the natives jars for the build
platform are picked automatically). It needs a Vulkan 1.2 driver with buffer device addresses and 64-bit shader
integers, which current AMD, NVIDIA and Intel drivers have. Allow LWJGL to call native code
with `--enable-native-access=org.lwjgl`.

```java
Device vulkan = Device.available().stream()
        .filter(d -> d.name().startsWith("vulkan")).findFirst().orElseThrow();
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

Element-wise operations in float32 and float64 are fused, as on OpenCL: a result is recorded
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
`brainiac.vulkan.statistics` (`true` prints the counts below at exit).
`VulkanBackend.status()` lists the devices found and `statistics()` counts dispatches,
barriers and submits. The examples take `--device vulkan` or `--device opencl` to pick one
GPU backend when both are present. CI runs the tests on lavapipe.

On an RX 9070 XT (2 epochs of MNIST; OpenCL in brackets): the MLP trains in 1.7 s (3.0 s)
and infers 600k samples/s (90-117k); the CNN trains in 3.2 s (3.1 s) and infers 130k
samples/s (158-227k). A 2048 x 2048 matmul reaches 0.9-1.1 TFLOP/s (1.8-1.9). Vulkan wins
where many small operations dominate; OpenCL's fused kernels win on large matmuls and
convolutions. With fusion, a long element-wise chain over 4M values runs at
2,572 M elements/s (1,237 without; OpenCL's fused chain 849), MNIST inference gains 35-60%,
and a transformer on Shakespeare trains 3 epochs in 10.8 s (OpenCL 17.5 s).

Tensors and models come in four number formats:

```java
Model precise = model.to(DType.FLOAT64);              // double precision
Model small = model.to(DType.BFLOAT16);               // half the memory
Trainer.of(model).mixedPrecision(DType.FLOAT16);      // compute in float16, float32 weights,
                                                      // dynamic loss scaling
```

Mixed types promote to the wider one, gradients keep their value's format, and saved
models keep each tensor's format. On the CPU, float32 uses the fast kernels and the other
formats are computed in double precision; on OpenCL, float64 needs `cl_khr_fp64` and
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
(`Tensor.packed`), and matrix products read the levels directly, on the CPU (SIMD), in
OpenCL and in Vulkan. That pays off when generating text one token at a time, which waits
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
