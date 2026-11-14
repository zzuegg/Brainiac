# Brainiac

A neural network library written from scratch in Java, with zero runtime dependencies.

Brainiac aims to cover everything from a tiny fully connected network to a full training
suite behind a small, easy public API. Every calculation goes through a pluggable
`Backend`, so the same model can later run on the GPU.

> Status: project skeleton. The first milestone (tensors, autograd, dense layers,
> training, data loading, XOR and Iris examples) is in progress.

## Requirements

- JDK 27 installed (the build uses a Gradle toolchain; Gradle itself runs on JDK 17+)
- Nothing else: the Gradle wrapper is included

## Build

```sh
./gradlew build                      # compile, test, javadoc
./gradlew :brainiac-examples:run     # run the example
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

Shared build configuration lives in `build-logic/` as Gradle convention plugins
(`brainiac.java-library`, `brainiac.java-application`). Each module has a
`module-info.java`; only public API packages are exported.

## Planned API

```java
var model = Model.sequential()
    .input(2)
    .dense(8, Activation.TANH)
    .dense(1, Activation.SIGMOID)
    .build();

model.trainer()
    .device(Device.best())          // GPU when a GPU backend is available, else CPU
    .loss(Loss.MSE)
    .optimizer(Optimizer.adam(0.01f))
    .epochs(2_000)
    .fit(data);

float[] out = model.predict(1, 0);
```
