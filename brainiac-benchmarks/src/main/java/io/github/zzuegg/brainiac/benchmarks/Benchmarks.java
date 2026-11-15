package io.github.zzuegg.brainiac.benchmarks;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;

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
     * @param args {@code --quick}, {@code --threads N}, {@code --filter TEXT}
     */
    public static void main(String[] args) {
        boolean quick = false;
        String filter = "";
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--quick" -> quick = true;
                case "--threads" -> System.setProperty("brainiac.cpu.threads", args[++i]);
                case "--filter" -> filter = args[++i];
                default -> throw new IllegalArgumentException("unknown option " + args[i]
                        + "; use --quick, --threads N or --filter TEXT");
            }
        }
        String threads = System.getProperty("brainiac.cpu.threads",
                String.valueOf(Runtime.getRuntime().availableProcessors()));
        System.out.printf(Locale.ROOT, "Brainiac benchmarks: Java %s, %s threads%n%n",
                Runtime.version(), threads);
        System.out.printf(Locale.ROOT, "%-36s %12s %16s%n", "case", "median", "throughput");

        for (Case c : cases(quick)) {
            if (!c.name.contains(filter)) {
                continue;
            }
            double seconds = measure(c.run, quick);
            System.out.printf(Locale.ROOT, "%-36s %9.3f ms %10.2f %s%n",
                    c.name, seconds * 1e3, c.work / seconds, c.unit);
        }
    }

    private static List<Case> cases(boolean quick) {
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
