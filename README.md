# Brainiac

A neural network library written from scratch in Java, with zero runtime dependencies.

Brainiac aims to cover everything from a tiny fully connected network to a full training
suite behind a small, easy public API. Every calculation goes through a pluggable
`Backend`, so the same model can later run on the GPU.

> Status: milestone 2 complete: on top of tensors, autograd, dense networks, training
> and data loading, there are dropout, batch norm, L2, learning-rate schedules, graph
> models with skip connections, and saving and loading models.

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
| `brainiac-tensor` | Shapes, float32 tensors, tensor math, and the `Backend` SPI |
| `brainiac-backend-cpu` | Default backend, plain Java on the CPU |
| `brainiac-autograd` | Reverse-mode automatic differentiation |
| `brainiac-nn` | Layers, activations, losses, initializers, models |
| `brainiac-train` | Optimizers, training loop, metrics, listeners |
| `brainiac-data` | Datasets, batching, splitting, CSV loading, preprocessing |
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

Saving and loading:

```java
model.save(Path.of("iris.brainiac"));
Model restored = Model.load(Path.of("iris.brainiac"));
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

Measure on your machine with:

```sh
./gradlew :brainiac-benchmarks:run                        # all cases
./gradlew :brainiac-benchmarks:run --args="--quick"       # shorter
./gradlew :brainiac-benchmarks:run --args="--threads 1"   # single-threaded
./gradlew :brainiac-benchmarks:run --args="--no-simd"     # without the Vector API
```
