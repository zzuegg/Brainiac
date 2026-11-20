# Working on Brainiac

Brainiac is a neural network library written from scratch in Java, plus Brainiac Studio, a local web
app on top of it. This file is for coding agents (Studio's own agents load it automatically).

## Build and test

- JDK 27 through a Gradle toolchain; Gradle itself runs on JDK 17 or newer. Build scripts use the
  Kotlin DSL (`build.gradle.kts`); shared settings live in `build-logic/`.
- `./gradlew build` compiles, tests and builds Javadoc for every module.
- One module: `./gradlew :brainiac-studio:test`. One class: add `--tests '*FileToolsTest'`.
- The compiler runs with `-Xlint:all -Werror`: every warning fails the build.
- To compile on an older JDK for a quick check: `-Pbrainiac.javaVersion=25`.

## Rules

- No external libraries in the AI/ML core (tensors, autograd, networks, training, inference, file
  formats). Tooling around it, such as Studio's servers or parsers, may use well-established ones.
- Every module has a `module-info.java`; only public API packages are exported.
- Public classes and methods have Javadoc. Write it plainly: what it does, not how.
- New behavior comes with tests (JUnit 5) in the module's `src/test/java`.

## Where things are

- `brainiac-tensor`, `brainiac-backend-*`: tensors and the CPU, Vulkan and CUDA backends.
- `brainiac-autograd`, `brainiac-nn`, `brainiac-train`, `brainiac-data`: networks and training.
- `brainiac-io`: JSON, safetensors, GGUF, tokenizers. `brainiac-llm`: language models and tool calling.
- `brainiac-audio`, `brainiac-diffusion`: speech, images and video.
- `brainiac-studio`: the web app. Agent tools are in
  `brainiac-studio/src/main/java/io/github/zzuegg/brainiac/studio/agent/` (`*Tools.java`), the web UI
  in `src/main/resources/io/github/zzuegg/brainiac/studio/web/`.

## Studio agent tools

- A tool is a public method marked `@Tools.Tool`, with `@Tier` (its risk), `@Returns` (its result's
  shape) and a `@Tools.Param` description per parameter. Its class carries `@ToolGroup`.
- Descriptions say what the tool does, when to use another tool instead, and its limits. Errors say
  what went wrong and what to try next.
- Give locations as a path relative to the project's folder and a line number.
- Renaming a tool: add the old name to `ToolNames.RENAMED` so saved roles and rules keep working.
