# Brainiac

A neural network library written from scratch in Java, with zero runtime dependencies.

Brainiac aims to cover everything from a tiny fully connected network to a full training
suite behind a small, easy public API. Every calculation goes through a pluggable
`Backend`, so the same model can later run on the GPU.

> Status: on top of tensors, autograd, dense networks, training and data loading, there
> are dropout, normalization, L2, learning-rate schedules, graph models, saving and
> loading, custom ops, 21 activations, and convolutional networks for 1-d, 2-d and 3-d
> data.

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
| `brainiac-autograd` | Reverse-mode automatic differentiation |
| `brainiac-nn` | Layers, activations, losses, initializers, models, the growing `Brain` |
| `brainiac-train` | Optimizers, training loop, metrics, listeners, `Plasticity` |
| `brainiac-data` | Datasets, batching, splitting, CSV loading, preprocessing |
| `brainiac-rl` | Reinforcement learning: environments, replay buffers, Q-learning, SARSA, REINFORCE, A2C, PPO, DQN |
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
input chain on the fly. This works for any model, without per-layer kernels, and the
generated code is plain OpenCL C 1.2, so it runs on every vendor's driver. Shapes and
strides are kernel arguments, so a training loop needs a few dozen generated kernels.
They compile in the background (the operations run one by one until then) and are cached
on disk, so later runs start at full speed. Results are the same with and without fusion.
Properties: `brainiac.opencl.fusion` (`false` turns it off), `brainiac.opencl.fusion.async`
(`false` compiles before running), `brainiac.opencl.cache` (directory, default
`~/.cache/brainiac/opencl`, or `none`).
Without an OpenCL GPU the backend reports itself unavailable and its tests are skipped;
CI runs them on PoCL.

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
