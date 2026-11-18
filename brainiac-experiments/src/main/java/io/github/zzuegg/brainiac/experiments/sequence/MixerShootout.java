package io.github.zzuegg.brainiac.experiments.sequence;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GatedDeltaNet;
import io.github.zzuegg.brainiac.experimental.nn.HyenaOperator;
import io.github.zzuegg.brainiac.experimental.nn.LinearAttention;
import io.github.zzuegg.brainiac.experimental.nn.MLstmLayer;
import io.github.zzuegg.brainiac.experimental.nn.Mamba2Layer;
import io.github.zzuegg.brainiac.experimental.nn.MambaLayer;
import io.github.zzuegg.brainiac.experimental.nn.Rwkv7TimeMix;
import io.github.zzuegg.brainiac.experimental.nn.S4dLayer;
import io.github.zzuegg.brainiac.experimental.nn.SLstmLayer;
import io.github.zzuegg.brainiac.experimental.nn.TestTimeTrainingLayer;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A shoot-out of sequence mixers on synthetic diagnostic tasks, in the spirit of the
 * "Zoology" and MAD synthetic benchmarks: every mixer is dropped, through its
 * {@link LayerSpec}, into the same tiny model (token plus previous-token embeddings, the
 * mixer, a readout at the last position), trained with the same budget, and scored on fresh
 * samples. The tasks isolate one capability each:
 *
 * <ul>
 *   <li>{@link #recall associative recall}: retrieve the value bound to a key, the capability
 *       that separates attention from fixed-state recurrences;</li>
 *   <li>{@link #recentRecall recent recall}: keys repeat and the latest binding wins, which
 *       needs order information (decay, gates or positions);</li>
 *   <li>{@link #parity parity}: a state-tracking task that linear time-invariant and
 *       diagonal non-negative recurrences cannot express, but rotary or non-linear
 *       recurrences can.</li>
 * </ul>
 *
 * <p>Papers: Arora et al., "Zoology: Measuring and Improving Recall in Efficient Language
 * Models", <a href="https://arxiv.org/abs/2312.04927">arXiv:2312.04927</a>; Poli et al.,
 * "Mechanistic Design and Scaling of Hybrid Architectures",
 * <a href="https://arxiv.org/abs/2403.17844">arXiv:2403.17844</a>; Grazzi et al., "Unlocking
 * State-Tracking in Linear RNNs Through Negative Eigenvalues",
 * <a href="https://arxiv.org/abs/2411.12537">arXiv:2411.12537</a>.
 */
public final class MixerShootout {

    private MixerShootout() {
    }

    /** Draws a batch of token rows and their targets. */
    @FunctionalInterface
    public interface Generator {
        /**
         * Draws a batch.
         *
         * @param random the random source
         * @param inputs filled with token ids, {@code [batch][length]}
         * @return the target class of each row
         */
        int[] draw(SplittableRandom random, int[][] inputs);
    }

    /**
     * A synthetic task.
     *
     * @param name a display name
     * @param vocabulary the number of token ids (also the number of answer classes)
     * @param length the sequence length
     * @param generator draws batches
     */
    public record Task(String name, int vocabulary, int length, Generator generator) {

        /**
         * Draws a batch.
         *
         * @param random the random source
         * @param size the batch size
         * @return the inputs {@code [size][length]} and, in {@code [1][size]} form, the targets
         */
        public int[][][] batch(SplittableRandom random, int size) {
            int[][] in = new int[size][length];
            int[] target = generator.draw(random, in);
            return new int[][][] {in, {target}};
        }
    }

    /**
     * One mixer's result.
     *
     * @param mixer the mixer's name
     * @param task the task's name
     * @param accuracy the accuracy on 256 fresh samples
     * @param parameters the mixer's parameter count
     * @param millis the training time
     */
    public record Result(String mixer, String task, double accuracy, long parameters, long millis) {
    }

    /**
     * Associative recall: {@code k1 v1 ... kn vn q}, answer the value bound to {@code q}.
     *
     * @param keys the number of key symbols (at least {@code pairs})
     * @param values the number of value symbols
     * @param pairs the bindings per sequence
     * @return the task
     */
    public static Task recall(int keys, int values, int pairs) {
        if (keys < pairs) {
            throw new IllegalArgumentException("need at least as many keys as pairs");
        }
        return new Task("recall(" + keys + "," + values + "," + pairs + ")", keys + values, 2 * pairs + 1,
                (random, in) -> {
                    int[] target = new int[in.length];
                    for (int b = 0; b < in.length; b++) {
                        int[] perm = new int[keys];
                        for (int i = 0; i < keys; i++) {
                            perm[i] = i;
                        }
                        for (int i = keys - 1; i > 0; i--) {
                            int j = random.nextInt(i + 1);
                            int tmp = perm[i];
                            perm[i] = perm[j];
                            perm[j] = tmp;
                        }
                        for (int p = 0; p < pairs; p++) {
                            in[b][2 * p] = perm[p];
                            in[b][2 * p + 1] = keys + random.nextInt(values);
                        }
                        int q = random.nextInt(pairs);
                        in[b][2 * pairs] = perm[q];
                        target[b] = in[b][2 * q + 1];
                    }
                    return target;
                });
    }

    /**
     * Recent recall: keys repeat, the answer is the value of the query's latest binding.
     *
     * @param keys the number of key symbols
     * @param values the number of value symbols
     * @param pairs the bindings per sequence
     * @return the task
     */
    public static Task recentRecall(int keys, int values, int pairs) {
        return new Task("recentRecall(" + keys + "," + values + "," + pairs + ")", keys + values, 2 * pairs + 1,
                (random, in) -> {
                    int[] target = new int[in.length];
                    for (int b = 0; b < in.length; b++) {
                        for (int p = 0; p < pairs; p++) {
                            in[b][2 * p] = random.nextInt(keys);
                            in[b][2 * p + 1] = keys + random.nextInt(values);
                        }
                        int q = in[b][2 * random.nextInt(pairs)];
                        in[b][2 * pairs] = q;
                        for (int p = pairs - 1; p >= 0; p--) {
                            if (in[b][2 * p] == q) {
                                target[b] = in[b][2 * p + 1];
                                break;
                            }
                        }
                    }
                    return target;
                });
    }

    /**
     * Parity of a random bit string.
     *
     * @param length the number of bits
     * @return the task
     */
    public static Task parity(int length) {
        return new Task("parity(" + length + ")", 2, length, (random, in) -> {
            int[] target = new int[in.length];
            for (int b = 0; b < in.length; b++) {
                for (int i = 0; i < length; i++) {
                    in[b][i] = random.nextInt(2);
                    target[b] ^= in[b][i];
                }
            }
            return target;
        });
    }

    /**
     * The recurrent and convolutional mixers of this theme, in their default sizes.
     *
     * @return name to spec, in a stable order
     */
    public static Map<String, LayerSpec> lineup() {
        Map<String, LayerSpec> m = new LinkedHashMap<>();
        m.put("linear attention", LinearAttention.of(2));
        m.put("S4D", S4dLayer.of(8));
        m.put("Mamba", MambaLayer.of(1, 8));
        m.put("Mamba-2", Mamba2Layer.of(2, 8, 8, 4, Mamba2Layer.Variant.REAL));
        m.put("Mamba-3 rotary", Mamba2Layer.of(2, 8, 8, 4, Mamba2Layer.Variant.ROTARY));
        m.put("Gated DeltaNet", GatedDeltaNet.of(2, GatedDeltaNet.Gating.SCALAR, 4));
        m.put("Kimi delta", GatedDeltaNet.of(2, GatedDeltaNet.Gating.CHANNEL, 4));
        m.put("RWKV-7", Rwkv7TimeMix.of(2, 4));
        m.put("mLSTM", MLstmLayer.of(2));
        m.put("sLSTM", SLstmLayer.of(16, 4));
        m.put("Hyena", HyenaOperator.of(2, 16));
        m.put("TTT-Linear", TestTimeTrainingLayer.of(2, TestTimeTrainingLayer.Variant.LINEAR, 4, 0.5f));
        return m;
    }

    /**
     * Trains one mixer on one task.
     *
     * @param name the mixer's display name
     * @param spec the mixer, built for input {@code [length, width]}
     * @param task the task
     * @param width the model width
     * @param steps Adam steps on batches of 32
     * @param lr the learning rate
     * @param seed the seed for weights and data
     * @return the result
     */
    public static Result run(String name, LayerSpec spec, Task task, int width, int steps, float lr, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        Layer mixer = spec.create(Shape.of(task.length(), width), random);
        Var embed = Var.param(Tensor.normal(Shape.of(task.vocabulary(), width), 0, 1, random));
        Var previous = Var.param(Tensor.normal(Shape.of(task.vocabulary(), width), 0, 1, random));
        Var readout = Var.param(Tensor.uniform(Shape.of(width, task.vocabulary()), -1, 1, random)
                .mul((float) Math.sqrt(3.0 / width)));
        List<Var> params = new ArrayList<>(List.of(embed, previous, readout));
        params.addAll(mixer.parameters());
        long count = 0;
        for (Var p : mixer.parameters()) {
            count += p.value().size();
        }
        Optimizer optimizer = Optimizer.adam(lr);
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            int[][][] batch = task.batch(random, 32);
            Var logits = logits(batch[0], embed, previous, readout, mixer, true, width);
            Functional.crossEntropy(logits, ids(batch[1][0])).backward();
            optimizer.step(params);
        }
        long millis = (System.nanoTime() - start) / 1_000_000;
        int[][][] test = task.batch(new SplittableRandom(seed + 1000), 256);
        float[] l = logits(test[0], embed, previous, readout, mixer, false, width).value().toArray();
        int classes = task.vocabulary();
        int correct = 0;
        for (int i = 0; i < test[1][0].length; i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (l[i * classes + c] > l[i * classes + best]) {
                    best = c;
                }
            }
            correct += best == test[1][0][i] ? 1 : 0;
        }
        return new Result(name, task.name(), correct / (double) test[1][0].length, count, millis);
    }

    /**
     * Trains one mixer like {@link #run(String, LayerSpec, Task, int, int, float, long)} (same
     * batches) and records the test accuracy after each checkpoint's number of steps, so the
     * value at a checkpoint equals a separate run of that budget.
     *
     * @param spec the mixer
     * @param task the task
     * @param width the model width
     * @param checkpoints increasing step counts
     * @param lr the learning rate
     * @param seed the seed for weights and data
     * @return the test accuracy at each checkpoint
     */
    public static double[] curve(LayerSpec spec, Task task, int width, int[] checkpoints, float lr, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        Layer mixer = spec.create(Shape.of(task.length(), width), random);
        Var embed = Var.param(Tensor.normal(Shape.of(task.vocabulary(), width), 0, 1, random));
        Var previous = Var.param(Tensor.normal(Shape.of(task.vocabulary(), width), 0, 1, random));
        Var readout = Var.param(Tensor.uniform(Shape.of(width, task.vocabulary()), -1, 1, random)
                .mul((float) Math.sqrt(3.0 / width)));
        List<Var> params = new ArrayList<>(List.of(embed, previous, readout));
        params.addAll(mixer.parameters());
        Optimizer optimizer = Optimizer.adam(lr);
        int[][][] test = task.batch(new SplittableRandom(seed + 1000), 256);
        double[] out = new double[checkpoints.length];
        int next = 0;
        int last = checkpoints[checkpoints.length - 1];
        for (int s = 0; s <= last; s++) {
            while (next < checkpoints.length && checkpoints[next] == s) {
                float[] l = Var.noGrad(() -> logits(test[0], embed, previous, readout, mixer, false, width)).value()
                        .toArray();
                out[next++] = accuracy(l, test[1][0], task.vocabulary());
            }
            if (s == last) {
                break;
            }
            int[][][] batch = task.batch(random, 32);
            Functional.crossEntropy(logits(batch[0], embed, previous, readout, mixer, true, width), ids(batch[1][0]))
                    .backward();
            optimizer.step(params);
        }
        return out;
    }

    private static double accuracy(float[] l, int[] targets, int classes) {
        int correct = 0;
        for (int i = 0; i < targets.length; i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (l[i * classes + c] > l[i * classes + best]) {
                    best = c;
                }
            }
            correct += best == targets[i] ? 1 : 0;
        }
        return correct / (double) targets.length;
    }

    /**
     * Trains every mixer of a lineup on every task.
     *
     * @param lineup name to spec
     * @param tasks the tasks
     * @param width the model width
     * @param steps Adam steps per run
     * @param lr the learning rate
     * @param seed the seed
     * @return one result per mixer and task, mixer-major
     */
    public static List<Result> run(Map<String, LayerSpec> lineup, List<Task> tasks, int width, int steps, float lr,
            long seed) {
        List<Result> out = new ArrayList<>();
        for (Map.Entry<String, LayerSpec> e : lineup.entrySet()) {
            for (Task task : tasks) {
                out.add(run(e.getKey(), e.getValue(), task, width, steps, lr, seed));
            }
        }
        return out;
    }

    /**
     * Formats results as a text table, one row per mixer and one accuracy column per task.
     *
     * @param results the results
     * @return the table
     */
    public static String table(List<Result> results) {
        List<String> tasks = new ArrayList<>();
        Map<String, Map<String, Result>> rows = new LinkedHashMap<>();
        for (Result r : results) {
            if (!tasks.contains(r.task())) {
                tasks.add(r.task());
            }
            rows.computeIfAbsent(r.mixer(), k -> new LinkedHashMap<>()).put(r.task(), r);
        }
        StringBuilder sb = new StringBuilder(String.format("%-18s %8s", "mixer", "params"));
        for (String t : tasks) {
            sb.append(String.format(" %22s", t));
        }
        sb.append('\n');
        for (Map.Entry<String, Map<String, Result>> row : rows.entrySet()) {
            Result any = row.getValue().values().iterator().next();
            sb.append(String.format("%-18s %8d", row.getKey(), any.parameters()));
            for (String t : tasks) {
                Result r = row.getValue().get(t);
                sb.append(r == null ? String.format(" %22s", "-")
                        : String.format(" %14.3f %5dms", r.accuracy(), r.millis()));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    private static Var logits(int[][] tokens, Var embed, Var previous, Var readout, Layer mixer, boolean training,
            int width) {
        int b = tokens.length;
        int t = tokens[0].length;
        float[] ids = new float[b * t];
        float[] prev = new float[b * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = tokens[i][j];
                prev[i * t + j] = tokens[i][Math.max(0, j - 1)];
            }
        }
        Var x = embed.gather(0, Tensor.of(Shape.of(b * t), ids))
                .add(previous.gather(0, Tensor.of(Shape.of(b * t), prev))).reshape(b, t, width);
        Var y = mixer.forward(x, training);
        return y.slice(1, t - 1, t).reshape(b, width).matmul(readout);
    }

    private static Tensor ids(int[] targets) {
        float[] f = new float[targets.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = targets[i];
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /**
     * Runs the full lineup on recall, recent recall and parity and prints the table.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        List<Task> tasks = List.of(recall(8, 8, 4), recentRecall(3, 4, 4), parity(6));
        System.out.print(table(run(lineup(), tasks, 16, 300, 0.01f, 1)));
    }
}
