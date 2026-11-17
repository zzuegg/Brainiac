package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/** A tiny shared training harness: token + previous-token embeddings, a mixer layer, a readout. */
final class MixerHarness {

    private MixerHarness() {
    }

    /** One batch: inputs {@code [b][t]} and the target class of the last position. */
    record Batch(int[][] inputs, int[] targets) {
    }

    /** Generates batches. */
    interface Task {
        Batch batch(SplittableRandom random, int size);

        int vocabulary();
    }

    /**
     * Associative recall: {@code k1 v1 k2 v2 ... q}, answer the value stored with {@code q}.
     * Keys are {@code 0..keys-1}, values {@code keys..keys+values-1}.
     */
    static Task recall(int keys, int values, int pairs) {
        return new Task() {
            @Override
            public Batch batch(SplittableRandom random, int size) {
                int[][] in = new int[size][2 * pairs + 1];
                int[] target = new int[size];
                for (int b = 0; b < size; b++) {
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
                    int[] vals = new int[pairs];
                    for (int p = 0; p < pairs; p++) {
                        vals[p] = keys + random.nextInt(values);
                        in[b][2 * p] = perm[p];
                        in[b][2 * p + 1] = vals[p];
                    }
                    int q = random.nextInt(pairs);
                    in[b][2 * pairs] = perm[q];
                    target[b] = vals[q];
                }
                return new Batch(in, target);
            }

            @Override
            public int vocabulary() {
                return keys + values;
            }
        };
    }

    /**
     * Recent recall: {@code pairs} key-value pairs where keys repeat (drawn from {@code keys}
     * symbols), then a query key; the answer is the value of the query's most recent pair.
     * Content alone cannot tell the occurrences apart, so this needs order information.
     */
    static Task recentRecall(int keys, int values, int pairs) {
        return new Task() {
            @Override
            public Batch batch(SplittableRandom random, int size) {
                int[][] in = new int[size][2 * pairs + 1];
                int[] target = new int[size];
                for (int b = 0; b < size; b++) {
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
                return new Batch(in, target);
            }

            @Override
            public int vocabulary() {
                return keys + values;
            }
        };
    }

    /** The model around a mixer. */
    static final class Harness {
        final Var embed;
        final Var previous;
        final Var readout;
        final Layer mixer;
        final int d;

        Harness(Layer mixer, int vocabulary, int d, SplittableRandom random) {
            this.mixer = mixer;
            this.d = d;
            this.embed = Var.param(Tensor.normal(Shape.of(vocabulary, d), 0, 1, random));
            this.previous = Var.param(Tensor.normal(Shape.of(vocabulary, d), 0, 1, random));
            this.readout = MixerOps.weight(random, d, vocabulary);
        }

        Var embed(int[][] tokens) {
            int b = tokens.length;
            int t = tokens[0].length;
            float[] ids = new float[b * t];
            float[] prev = new float[b * t];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < t; j++) {
                    ids[i * t + j] = tokens[i][j];
                    prev[i * t + j] = j == 0 ? tokens[i][0] : tokens[i][j - 1];
                }
            }
            Var x = embed.gather(0, Tensor.of(Shape.of(b * t), ids));
            Var p = previous.gather(0, Tensor.of(Shape.of(b * t), prev));
            return x.add(p).reshape(b, t, d);
        }

        Var logits(int[][] tokens, Function<Var, Var> forward) {
            Var x = embed(tokens);
            Var y = forward.apply(x);
            int t = tokens[0].length;
            Var last = y.slice(1, t - 1, t).reshape(tokens.length, d);
            return last.matmul(readout);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embed, previous, readout));
            p.addAll(mixer.parameters());
            return p;
        }
    }

    static Tensor ids(int[] targets) {
        float[] f = new float[targets.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = targets[i];
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    static double accuracy(Var logits, int[] targets) {
        float[] l = logits.value().toArray();
        int classes = logits.shape().dim(1);
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
     * Trains a mixer on a task and returns the accuracy on fresh batches.
     *
     * @param extraLoss an extra loss term read after each forward (for auxiliary losses), or null
     */
    static double train(Layer mixer, Task task, int d, int steps, float lr, long seed,
            java.util.function.Supplier<Var> extraLoss) {
        return train(mixer, task, d, steps, lr, seed, extraLoss, null);
    }

    static double train(Layer mixer, Task task, int d, int steps, float lr, long seed,
            java.util.function.Supplier<Var> extraLoss, Runnable afterStep) {
        SplittableRandom random = new SplittableRandom(seed);
        Harness h = new Harness(mixer, task.vocabulary(), d, random);
        Optimizer opt = Optimizer.adam(lr);
        List<Var> params = h.parameters();
        for (int s = 0; s < steps; s++) {
            Batch batch = task.batch(random, 32);
            Var loss = Functional.crossEntropy(h.logits(batch.inputs(), x -> mixer.forward(x, true)), ids(batch.targets()));
            if (extraLoss != null) {
                Var extra = extraLoss.get();
                if (extra != null) {
                    loss = loss.add(extra);
                }
            }
            loss.backward();
            opt.step(params);
            if (afterStep != null) {
                afterStep.run();
            }
        }
        Batch test = task.batch(new SplittableRandom(seed + 1000), 256);
        return accuracy(h.logits(test.inputs(), mixer::forward), test.targets());
    }

    static double train(Layer mixer, Task task, int d, int steps, float lr, long seed) {
        return train(mixer, task, d, steps, lr, seed, null);
    }
}
