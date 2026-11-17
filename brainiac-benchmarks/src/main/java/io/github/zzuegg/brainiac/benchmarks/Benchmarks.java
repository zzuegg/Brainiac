package io.github.zzuegg.brainiac.benchmarks;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.KvCache;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Quantization;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Measures the speed of core operations and of a training loop.
 *
 * <p>There are no benchmark libraries in this project, so this harness keeps things
 * simple: each case is warmed up until the JIT compiler has settled, then timed several
 * times, and the median is reported.
 *
 * <pre>
 * ./gradlew :brainiac-benchmarks:run                          # all cases
 * ./gradlew :brainiac-benchmarks:run --args="--quick"         # fewer, shorter cases
 * ./gradlew :brainiac-benchmarks:run --args="--threads 1"     # single-threaded backend
 * ./gradlew :brainiac-benchmarks:run --args="--filter matmul" # cases whose name contains "matmul"
 * ./gradlew :brainiac-benchmarks:run --args="--no-simd"      # plain Java kernels only
 * ./gradlew :brainiac-benchmarks:run --args="--device vulkan --filter matvec"  # products on a GPU
 * </pre>
 */
public final class Benchmarks {

    private Benchmarks() {
    }

    /** One timed case: {@code run} does the work once, {@code work} is its size in units. */
    private record Case(String name, double work, String unit, Runnable run) {
    }

    /**
     * Runs the benchmarks.
     *
     * @param args {@code --quick}, {@code --threads N}, {@code --filter TEXT}, {@code --no-simd},
     *        {@code --device NAME} (the start of a device name, such as {@code cpu} or
     *        {@code vulkan}: the matrix-vector and text generation cases run there)
     */
    public static void main(String[] args) {
        boolean quick = false;
        Device device = Device.cpu();
        String filter = "";
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--quick" -> quick = true;
                case "--threads" -> System.setProperty("brainiac.cpu.threads", args[++i]);
                case "--filter" -> filter = args[++i];
                case "--no-simd" -> System.setProperty("brainiac.cpu.simd", "false");
                case "--device" -> {
                    String name = args[++i];
                    device = Device.available().stream().filter(d -> d.name().startsWith(name)).findFirst()
                            .orElseThrow(() -> new IllegalArgumentException("no device " + name + "; available: "
                                    + Device.available()));
                }
                default -> throw new IllegalArgumentException("unknown option " + args[i]
                        + "; use --quick, --threads N, --filter TEXT, --no-simd or --device NAME");
            }
        }
        String threads = System.getProperty("brainiac.cpu.threads",
                String.valueOf(Runtime.getRuntime().availableProcessors()));
        boolean simd = ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent()
                && Boolean.parseBoolean(System.getProperty("brainiac.cpu.simd", "true"));
        System.out.printf(Locale.ROOT, "Brainiac benchmarks: Java %s, %s threads, SIMD %s%s%n%n",
                Runtime.version(), threads, simd ? "on" : "off",
                device.kind() == Device.Kind.CPU ? "" : ", products on " + device);
        System.out.printf(Locale.ROOT, "%-36s %12s %16s%n", "case", "median", "throughput");

        for (Case c : cases(quick, device)) {
            if (!c.name.contains(filter)) {
                continue;
            }
            double seconds = measure(c.run, quick);
            System.out.printf(Locale.ROOT, "%-36s %9.3f ms %10.2f %s%n",
                    c.name, seconds * 1e3, c.work / seconds, c.unit);
        }
    }

    private static List<Case> cases(boolean quick, Device device) {
        SplittableRandom random = new SplittableRandom(42);
        List<Case> cases = new ArrayList<>();
        int[] sizes = quick ? new int[] {128, 512} : new int[] {64, 128, 256, 512, 1024};
        for (int n : sizes) {
            Tensor a = Tensor.uniform(Shape.of(n, n), -1, 1, random);
            Tensor b = Tensor.uniform(Shape.of(n, n), -1, 1, random);
            cases.add(new Case("matmul " + n + "x" + n, 2.0 * n * n * n / 1e9, "GFLOP/s", () -> a.matmul(b)));
        }
        Tensor x = Tensor.uniform(Shape.of(256, 784), -1, 1, random);
        Tensor w = Tensor.uniform(Shape.of(784, 256), -1, 1, random);
        cases.add(new Case("matmul 256x784 * 784x256", 2.0 * 256 * 784 * 256 / 1e9, "GFLOP/s", () -> x.matmul(w)));

        // one token at a time through a large layer: reading the weights is the cost
        int d = quick ? 2048 : 4096;
        // on a GPU, each run queues `repeat` products and waits for the last
        int repeat = device.kind() == Device.Kind.CPU ? 1 : 50;
        Tensor big = Tensor.uniform(Shape.of(d, d), -1, 1, random).to(device);
        for (int rows : new int[] {1, 8}) {
            Tensor v = Tensor.uniform(Shape.of(rows, d), -1, 1, random).to(device);
            double flops = 2.0 * rows * d * d * repeat / 1e9;
            cases.add(new Case("matvec " + rows + "x" + d + " f32 weights", flops, "GFLOP/s",
                    () -> products(v, big, repeat)));
            for (int bits : new int[] {8, 4}) {
                Tensor packed = Tensor.packed(packed(d, d, bits, random), Device.cpu()).to(device);
                cases.add(new Case("matvec " + rows + "x" + d + " int" + bits + " weights", flops, "GFLOP/s",
                        () -> products(v, packed, repeat)));
            }
        }

        // writing text one token at a time with a GPT-2-small-sized transformer (6 blocks of 768)
        int width = quick ? 384 : 768;
        Model.Builder gptBuilder = Model.sequential().input(128).add(Embedding.of(256, width));
        for (int i = 0; i < 6; i++) {
            gptBuilder.add(TransformerBlock.of(width / 64).causal(true).rotary(10000).ffnSize(4 * width));
        }
        Model lm = gptBuilder.layer(LayerNorm.of(width)).dense(256, Activation.SOFTMAX).seed(1).build().to(device);
        int tokens = 16;
        cases.add(new Case("generate " + tokens + " tokens, 6x" + width + " f32", tokens, "tokens/s",
                () -> generate(lm, tokens)));
        for (Quantization q : List.of(Quantization.int8(), Quantization.int4().groups(32))) {
            Model packedLm = lm.copy(device).quantize(q).pack();
            cases.add(new Case("generate " + tokens + " tokens, 6x" + width + " " + q.bits() + "-bit packed", tokens,
                    "tokens/s", () -> generate(packedLm, tokens)));
        }

        int elements = 1 << 22;
        Tensor u = Tensor.uniform(Shape.of(elements), -1, 1, random);
        Tensor v = Tensor.uniform(Shape.of(elements), -1, 1, random);
        cases.add(new Case("add 4M", elements / 1e6, "M elem/s", () -> u.add(v)));
        cases.add(new Case("tanh 4M", elements / 1e6, "M elem/s", u::tanh));
        cases.add(new Case("sum 4M", elements / 1e6, "M elem/s", u::sum));
        Tensor rows = u.reshape(4096, 1024);
        Tensor bias = Tensor.uniform(Shape.of(1024), -1, 1, random);
        cases.add(new Case("add bias 4096x1024", elements / 1e6, "M elem/s", () -> rows.add(bias)));
        cases.add(new Case("sum axis 0 4096x1024", elements / 1e6, "M elem/s", () -> rows.sum(0)));
        cases.add(new Case("transpose 4096x1024", elements / 1e6, "M elem/s", rows::transpose));

        // activations, forward and backward: fused kernels against the same function
        // written as a chain of Var operations
        Var in = Var.param(rows);
        Map<String, UnaryOperator<Var>> composed = new LinkedHashMap<>();
        composed.put("gelu", t -> t.mul(t.mul((float) (1 / Math.sqrt(2))).erf().add(1)).mul(0.5f));
        composed.put("silu", t -> t.mul(t.sigmoid()));
        composed.put("leaky_relu", t -> t.relu().sub(t.neg().relu().mul(0.01f)));
        composed.put("hard_swish", t -> t.mul(t.div(6).add(0.5f).clamp(0, 1)));
        Map<String, Activation> fused = Map.of("gelu", Activation.GELU, "silu", Activation.SILU,
                "leaky_relu", Activation.leakyRelu(), "hard_swish", Activation.HARD_SWISH);
        for (var e : composed.entrySet()) {
            Activation a = fused.get(e.getKey());
            UnaryOperator<Var> f = e.getValue();
            cases.add(new Case(e.getKey() + " fwd+bwd 4M fused", elements / 1e6, "M elem/s",
                    () -> activationStep(in, a::forward)));
            cases.add(new Case(e.getKey() + " fwd+bwd 4M composed", elements / 1e6, "M elem/s",
                    () -> activationStep(in, f)));
        }

        int samples = quick ? 1024 : 4096;
        float[][] features = new float[samples][784];
        float[][] labels = new float[samples][10];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < 784; j++) {
                features[i][j] = (float) random.nextDouble();
            }
            labels[i][random.nextInt(10)] = 1;
        }
        Dataset data = Dataset.of(features, labels);
        Model model = Model.sequential().input(784)
                .dense(256, Activation.RELU).dense(128, Activation.RELU).dense(10, Activation.SOFTMAX)
                .seed(1).build();
        Trainer trainer = Trainer.of(model).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.001f))
                .batchSize(128).epochs(1).seed(1);
        cases.add(new Case("train MLP 784-256-128-10, 1 epoch", samples / 1e3, "k samples/s", () -> trainer.fit(data)));
        return cases;
    }

    /**
     * Multiplies {@code repeat} times in a chain, each product by the previous one's result,
     * and waits for the last: a backend that defers work then has to do every product.
     */
    private static void products(Tensor v, Tensor w, int repeat) {
        Tensor out = v;
        for (int i = 0; i < repeat; i++) {
            out = out.matmul(w);
        }
        if (repeat > 1) {
            out.toArray();
        }
    }

    /** Writes tokens one at a time through a fresh cache. */
    private static void generate(Model model, int tokens) {
        KvCache cache = KvCache.of(model);
        Tensor out = cache.append(1, 2, 3);
        for (int i = 0; i < tokens; i++) {
            out.toArray();
            out = cache.append(i % 200);
        }
    }

    /** Random levels with one scale per column. */
    private static PackedMatrix packed(int rows, int cols, int bits, SplittableRandom random) {
        int limit = (1 << (bits - 1)) - 1;
        byte[] levels = new byte[rows * cols];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) random.nextInt(-limit, limit + 1);
        }
        float[] scales = new float[cols];
        Arrays.fill(scales, 1f / limit);
        return PackedMatrix.of(levels, scales, rows, cols, bits, rows);
    }

    private static void activationStep(Var in, UnaryOperator<Var> f) {
        in.zeroGrad();
        f.apply(in).backward(ONES_4M);
    }

    private static final Tensor ONES_4M = Tensor.ones(Shape.of(4096, 1024));

    private static double measure(Runnable run, boolean quick) {
        // warm up for a fixed time so the JIT compiler has compiled the hot loops
        long warmupNanos = quick ? 300_000_000L : 1_000_000_000L;
        long start = System.nanoTime();
        int warmups = 0;
        while (System.nanoTime() - start < warmupNanos || warmups < 3) {
            run.run();
            warmups++;
        }
        int samples = quick ? 5 : 11;
        double[] times = new double[samples];
        for (int i = 0; i < samples; i++) {
            long t0 = System.nanoTime();
            run.run();
            times[i] = (System.nanoTime() - t0) / 1e9;
        }
        Arrays.sort(times);
        return times[samples / 2];
    }
}
