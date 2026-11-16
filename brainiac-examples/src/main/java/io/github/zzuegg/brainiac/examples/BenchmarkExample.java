package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.SequencePool;
import io.github.zzuegg.brainiac.nn.SwitchingDense;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Kind;
import io.github.zzuegg.brainiac.train.Benchmark.Report;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Progress;
import io.github.zzuegg.brainiac.train.Trainer;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Runs models on the benchmark {@link Tasks} from the command line:
 *
 * <pre>
 * ./gradlew :brainiac-examples:run --args="bench mnist cnn lenet mlp --epochs 2"
 * ./gradlew :brainiac-examples:run --args="bench offline"
 * ./gradlew :brainiac-examples:run --args="bench mnist path/to/model.brainiac"
 * </pre>
 *
 * <p>Models are named from {@link #MODELS} or given as a path to a saved model file, which
 * is tested as it is (add {@code --train} to train it further first). From Java, build any
 * model and call {@link Benchmark#run(Task, String, Model)} with a task from {@link Tasks}.
 */
public final class BenchmarkExample {

    private BenchmarkExample() {
    }

    /** Built-in model names; not every model fits every task. */
    public static final List<String> MODELS =
            List.of("default", "linear", "mlp", "switching", "cnn", "lenet", "resnet", "bilstm", "gru", "transformer");

    /**
     * Builds a named model for a task.
     *
     * @param name one of {@link #MODELS}
     * @param task the task the model is for
     * @return an untrained model
     * @throws IllegalArgumentException if the name is unknown or does not fit the task
     */
    public static Model model(String name, Task task) {
        Shape in = task.inputShape();
        Shape out = task.outputShape();
        return switch (name) {
            case "default" -> defaultModel(task);
            case "linear" -> dense(task, 0);
            case "mlp" -> dense(task, 2);
            case "switching" -> switching(task, 2);
            case "cnn" -> {
                requireImages(name, task);
                yield cnn(in, out);
            }
            case "resnet" -> {
                requireImages(name, task);
                yield CifarExample.resnet(in.dim(0), in.dim(1), Math.toIntExact(out.size()), 3);
            }
            case "lenet" -> {
                requireImages(name, task);
                yield lenet(in, out);
            }
            case "bilstm" -> recurrent(task, Recurrent.lstm(64).returnSequences(true)
                    .bidirectional(Recurrent.Merge.CONCAT), true);
            case "gru" -> recurrent(task, Recurrent.gru(64).returnSequences(true), false);
            case "transformer" -> transformer(task);
            default -> throw new IllegalArgumentException("unknown model " + name + ", try one of " + MODELS
                    + " or a path to a saved model");
        };
    }

    /**
     * The model the task's own example uses.
     *
     * @param task the task
     * @return an untrained model
     */
    public static Model defaultModel(Task task) {
        // every tokenizer of shakespeare gets the same model, sized to its vocabulary
        return switch (task.name().startsWith("shakespeare/") ? "shakespeare" : task.name()) {
            case "mnist", "fashion" -> MnistExample.model(28, 10);
            case "cifar10" -> CifarExample.resnet(3, 32, 10, 3);
            case "shakespeare" -> LanguageExample.model(task.outputShape().dim(1), task.inputShape().dim(0), 128, 4, 4);
            case "shapes" -> MnistExample.model(16, 3);
            case "waves" -> SequenceExample.forecaster();
            case "reversal" -> SequenceExample.reverser();
            case "sentiment" -> SentimentExample.transformer();
            case "spiral" -> SpiralExample.model(3);
            case "xor" -> XorExample.model();
            default -> dense(task, 1);
        };
    }

    /**
     * A plain network: flatten, {@code hidden} layers of 128 ReLU units, then the output.
     *
     * @param task the task
     * @param hidden the number of hidden layers; 0 gives a linear (softmax regression) model
     * @return an untrained model
     */
    public static Model dense(Task task, int hidden) {
        Model.Builder b = Model.sequential().input(task.inputShape().toArray()).flatten();
        int width = task.inputShape().size() <= 8 ? 32 : 128;
        for (int i = 0; i < hidden; i++) {
            b.dense(width, Activation.RELU);
        }
        return head(b.dense(Math.toIntExact(task.outputShape().size())), task).seed(42).build();
    }

    /**
     * Like {@link #dense}, but in every hidden layer each unit's inputs choose its activation
     * among relu, tanh, gelu and sigmoid ({@link SwitchingDense}).
     *
     * @param task the task
     * @param hidden the number of hidden layers
     * @return an untrained model
     */
    public static Model switching(Task task, int hidden) {
        Model.Builder b = Model.sequential().input(task.inputShape().toArray()).flatten();
        int width = task.inputShape().size() <= 8 ? 32 : 128;
        for (int i = 0; i < hidden; i++) {
            b.add(SwitchingDense.of(width, Activation.RELU, Activation.TANH, Activation.GELU, Activation.SIGMOID));
        }
        return head(b.dense(Math.toIntExact(task.outputShape().size())), task).seed(42).build();
    }

    private static Model cnn(Shape in, Shape out) {
        return MnistExample.model(in.dim(0), in.dim(1), Math.toIntExact(out.size()));
    }

    private static Model lenet(Shape in, Shape out) {
        return Model.sequential()
                .input(in.toArray())
                .add(Conv.of(6, 5, 5).padding(Padding.SAME).activation(Activation.TANH))
                .avgPool(2, 2)
                .add(Conv.of(16, 5, 5).activation(Activation.TANH))
                .avgPool(2, 2)
                .flatten()
                .dense(120, Activation.TANH)
                .dense(84, Activation.TANH)
                .dense(Math.toIntExact(out.size()), Activation.SOFTMAX)
                .seed(42)
                .build();
    }

    private static Model recurrent(Task task, Recurrent.Spec layer, boolean wholeSequence) {
        Shape in = sequenceInput(task);
        Model.Builder b = sequenceStart(task).add(layer);
        if (task.outputShape().rank() == 2 && task.outputShape().dim(0) == in.dim(0)) {
            b.dense(task.outputShape().dim(1));     // one output per step
        } else {
            b.sequencePool(wholeSequence ? SequencePool.Kind.MEAN : SequencePool.Kind.LAST)
                    .dense(Math.toIntExact(task.outputShape().size()));
        }
        return head(b, task).seed(42).build();
    }

    private static Model transformer(Task task) {
        Shape in = sequenceInput(task);
        Model.Builder b = sequenceStart(task)
                .dense(32)
                .positionalEncoding()
                .add(TransformerBlock.of(4).ffnSize(64))
                .add(TransformerBlock.of(4).ffnSize(64));
        if (task.outputShape().rank() == 2 && task.outputShape().dim(0) == in.dim(0)) {
            b.dense(task.outputShape().dim(1));
        } else {
            b.sequencePool(SequencePool.Kind.MEAN).dense(Math.toIntExact(task.outputShape().size()));
        }
        return head(b, task).seed(42).build();
    }

    /** Token tasks: their number of word ids. Their sequence models start with an embedding. */
    private static final java.util.Map<String, Integer> VOCABULARIES =
            java.util.Map.of("sentiment", SentimentExample.VOCABULARY);

    private static Model.Builder sequenceStart(Task task) {
        Model.Builder b = Model.sequential().input(task.inputShape().toArray());
        Integer vocabulary = VOCABULARIES.get(task.name());
        return vocabulary != null ? b.add(Embedding.of(vocabulary, 32).paddingIndex(0)) : b;
    }

    private static Shape sequenceInput(Task task) {
        Shape in = task.inputShape();
        if (VOCABULARIES.containsKey(task.name())) {
            return Shape.of(in.dim(0), 32);         // token ids, embedded into 32 values each
        }
        if (in.rank() == 2) {
            return in;
        }
        if (in.rank() == 3 && in.dim(0) == 1) {
            return Shape.of(in.dim(1), in.dim(2));    // an image read row by row
        }
        throw new IllegalArgumentException("sequence models need [steps, features] input; task " + task.name()
                + " has " + in);
    }

    private static void requireImages(String model, Task task) {
        if (task.inputShape().rank() != 3) {
            throw new IllegalArgumentException(model + " needs image input [channels, height, width]; task "
                    + task.name() + " has " + task.inputShape());
        }
    }

    private static Model.Builder head(Model.Builder b, Task task) {
        Shape out = task.outputShape();
        if (out.rank() > 1) {
            b.reshape(out.toArray());
        }
        if (task.kind() == Kind.CLASSIFICATION) {
            b.activation(out.dim(out.rank() - 1) == 1 ? Activation.SIGMOID : Activation.SOFTMAX);
        }
        return b;
    }

    /**
     * Runs the benchmark.
     *
     * @param args a task name, {@code offline} or {@code all}; then model names or files;
     *        then {@code --epochs N}, {@code --limit N} (training samples), {@code --train},
     *        {@code --quiet}, {@code --confusion}, {@code --device cpu|gpu|best},
     *        {@code --dtype f64|f16|bf16}, {@code --mixed f16|bf16},
     *        {@code --precisions f32,bf16,f16} (one run per precision, to compare),
     *        {@code --tokenizers char,bpe:1024} (for shakespeare: one task per tokenizer),
     *        {@code --cache none,restart,sliding,sinks:4,h2o:32,restart/bf16} (for shakespeare:
     *        compare KV cache strategies on long text)
     */
    public static void main(String[] args) {
        if (args.length == 0) {
            usage();
            return;
        }
        List<String> models = new ArrayList<>();
        Integer epochs = null;
        int limit = Integer.MAX_VALUE;
        boolean trainLoaded = false;
        boolean quiet = false;
        boolean confusion = false;
        Device device = Device.getDefault();
        DType dtype = DType.FLOAT32;
        DType mixed = null;
        List<DType> precisions = null;
        List<String> tokenizers = null;
        for (int i = 1; i < args.length; i++) {
            if (args[i].equals("--cache")) {
                cacheStrategies = List.of(args[++i].split(","));
                continue;
            }
            switch (args[i]) {
                case "--device" -> device = switch (args[++i]) {
                    case "cpu" -> Device.cpu();
                    case "gpu" -> Device.gpu();
                    case "best" -> Device.best();
                    default -> throw new IllegalArgumentException("--device takes cpu, gpu or best");
                };
                case "--dtype" -> dtype = dtype(args[++i]);
                case "--mixed" -> mixed = dtype(args[++i]);
                case "--precisions" -> {
                    precisions = new ArrayList<>();
                    for (String p : args[++i].split(",")) {
                        DType type = dtype(p);
                        precisions.add(type == DType.FLOAT32 ? null : type);
                    }
                }
                case "--tokenizers", "--tokenizer" -> tokenizers = List.of(args[++i].split(","));
                case "--epochs" -> epochs = Integer.parseInt(args[++i]);
                case "--limit" -> limit = Integer.parseInt(args[++i]);
                case "--train" -> trainLoaded = true;
                case "--quiet" -> quiet = true;
                case "--confusion" -> confusion = true;
                default -> models.add(args[i]);
            }
        }
        if (models.isEmpty()) {
            models.add("default");
        }
        List<Task> tasks = switch (args[0]) {
            case "all" -> Tasks.all();
            case "offline" -> Tasks.offline();
            case "shakespeare" -> tokenizers == null ? List.of(Tasks.shakespeare())
                    : tokenizers.stream().map(Tasks::shakespeare).toList();
            default -> List.of(Tasks.get(args[0]));
        };
        if (precisions == null) {
            precisions = new ArrayList<>();
            precisions.add(mixed);
        }
        boolean comparing = precisions.size() > 1;
        System.out.println("device " + device + (dtype != DType.FLOAT32 ? ", " + dtype : "")
                + (comparing ? ", comparing " + precisions.stream()
                        .map(BenchmarkExample::precisionName).toList()
                        : mixed != null ? ", mixed precision " + mixed : ""));
        List<Report> reports = new ArrayList<>();
        for (Task task : tasks) {
            Task t = limit == Integer.MAX_VALUE ? task : task.limit(limit, Integer.MAX_VALUE);
            System.out.printf("%n== %s: %s (%,d train, %,d test)%n", t.name(), t.description(), t.train().size(),
                    t.test().size());
            for (String name : models) {
                for (DType precision : precisions) {
                    String label = !comparing ? name
                            : name + " " + precisionName(precision);
                    Report r = runOne(t, name, label, new Options(epochs, !quiet, device, dtype, precision),
                            trainLoaded);
                    if (r != null) {
                        if (confusion) {
                            System.out.print(r.confusionTable());
                        }
                        reports.add(r);
                    }
                }
            }
        }
        System.out.println();
        System.out.print(Benchmark.table(reports));
    }

    /** KV cache strategies to compare after a language model is trained ({@code --cache}). */
    private static List<String> cacheStrategies = List.of();

    /** Reads 2,000 test tokens through each cache strategy, for loss on long text and speed. */
    private static void compareCaches(Model model, io.github.zzuegg.brainiac.data.Tokenizer tokenizer, String text) {
        if (cacheStrategies.isEmpty()) {
            return;
        }
        String test = text.substring((int) (text.length() * 0.9));
        System.out.println("reading 2,000 test tokens one at a time (window " + model.inputShape().dim(0) + "):");
        System.out.printf("  %-18s %12s %10s%n", "cache", "loss/char", "tokens/s");
        for (String spec : cacheStrategies) {
            String[] parts = spec.split("/");
            io.github.zzuegg.brainiac.nn.KvCache.Strategy strategy = parts[0].equals("none") ? null
                    : io.github.zzuegg.brainiac.nn.KvCache.Strategy.parse(parts[0]);
            DType storage = parts.length > 1 ? dtype(parts[1]) : null;
            double[] r = LanguageExample.streamLoss(strategy, storage, model, tokenizer, test, 2000);
            System.out.printf("  %-18s %12.4f %10.0f%n", spec, r[0], r[1]);
        }
    }

    /** Trains (or loads) and tests one model; null if the named model does not fit the task. */
    private static Report runOne(Task t, String name, String label, Options options, boolean trainLoaded) {
        Report r;
        Model model;
        Path file = Path.of(name);
        if (Files.isRegularFile(file)) {
            model = options.prepare(load(file));
            String fileLabel = file.getFileName() + label.substring(name.length());
            r = trainLoaded ? Benchmark.run(t, fileLabel, model, options.settings())
                    : Benchmark.evaluate(t, fileLabel, model);
        } else {
            try {
                model = options.prepare(model(name, t));
            } catch (IllegalArgumentException e) {
                System.out.println("skipping " + name + ": " + e.getMessage());
                return null;
            }
            r = Benchmark.run(t, label, model, options.settings());
        }
        System.out.print(r);
        if (t.name().startsWith("shakespeare")) {
            String spec = t.name().contains("/") ? t.name().substring(t.name().indexOf('/') + 1) : "char";
            io.github.zzuegg.brainiac.data.Tokenizer tokenizer;
            String text;
            try {
                text = LanguageExample.load(LanguageExample.DATA_FILE);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
            tokenizer = LanguageExample.tokenizer(spec, text);
            System.out.println("sample:\n" + LanguageExample.generate(model, tokenizer, "ROMEO:\n",
                    (int) Math.ceil(400 / tokenizer.charactersPerToken("ROMEO:\nWhat light through yonder window "
                            + "breaks? It is the east, and Juliet is the sun.")), 0.8f, 1) + "\n");
            compareCaches(model, tokenizer, text);
        }
        return r;
    }

    /** How models are placed and trained, from the command line. */
    private record Options(Integer epochs, boolean verbose, Device device, DType dtype, DType mixed) {

        Model prepare(Model model) {
            return model.to(device).to(dtype);
        }

        Consumer<Trainer> settings() {
            return t -> {
                if (epochs != null) {
                    t.epochs(epochs);
                }
                if (mixed != null) {
                    t.mixedPrecision(mixed);
                }
                if (verbose) {
                    t.listener(Progress.every(Math.max(1, t.epochs() / 10)));
                }
            };
        }
    }

    /** @return f32 for plain training, else the mixed-precision type, such as bf16-mixed */
    private static String precisionName(DType mixed) {
        return mixed == null ? "f32" : (mixed == DType.BFLOAT16 ? "bf16" : mixed == DType.FLOAT16 ? "f16"
                : mixed.name().toLowerCase()) + "-mixed";
    }

    private static DType dtype(String name) {
        return switch (name) {
            case "f32", "float32" -> DType.FLOAT32;
            case "f64", "float64" -> DType.FLOAT64;
            case "f16", "float16" -> DType.FLOAT16;
            case "bf16", "bfloat16" -> DType.BFLOAT16;
            default -> throw new IllegalArgumentException("unknown number format " + name + ", try f32, f64, f16, bf16");
        };
    }

    private static Model load(Path file) {
        try {
            return Model.load(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static void usage() {
        System.out.println("""
                Usage: ./gradlew :brainiac-examples:run --args="bench <task> [model...] [options]"
                  task     %s, offline (all without download) or all
                  model    %s, or a saved model file
                  --epochs N    train for N epochs instead of the task's default
                  --limit N     train on at most N samples
                  --train       train a loaded model further before testing it
                  --confusion   print the confusion matrix
                  --quiet       no per-epoch progress
                  --device D    cpu, gpu (OpenCL) or best
                  --dtype T     run the model in f64, f16 or bf16 instead of f32
                  --mixed T     train in mixed precision: f16 or bf16 compute, f32 weights
                  --precisions L  train each model once per precision, e.g. f32,bf16,f16
                                (f16 and bf16 mean mixed precision), to compare them
                  --tokenizers L  for shakespeare, one run per tokenizer, e.g.
                                char,byte,word:2048,bpe:1024,wordpiece:1024,unigram:1024
                  --cache L     for shakespeare, read 2,000 test tokens through each KV cache
                                strategy: none (recompute), restart, sliding, sinks:4, h2o:32;
                                add /bf16 or /f16 to store keys and values narrower""".formatted(Tasks.NAMES, MODELS));
    }
}
