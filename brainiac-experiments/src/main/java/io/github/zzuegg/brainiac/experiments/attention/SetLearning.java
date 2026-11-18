package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SetEncoder;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.11: learning functions of sets. Three tasks, trained on sets of 3 to 8 elements
 * and tested on 8 (in distribution) and 12 (larger than any training set):
 *
 * <ul>
 *   <li>{@link Task#MAX}: the maximum of scalars in {@code [0, 1]} (mean squared error);</li>
 *   <li>{@link Task#UNIQUE}: the number of distinct tokens among one-hot tokens from 6 values
 *       (accuracy of the rounded regression);</li>
 *   <li>{@link Task#CLUSTERS}: amortised clustering, the two centres of a two-Gaussian 2-D point
 *       set, trained with a permutation-min loss (mean centre distance).</li>
 * </ul>
 *
 * Compared: {@link SetEncoder} as Deep Sets and as a Set Transformer, against an MLP on the
 * sorted elements zero-padded to 12 slots with a presence flag (the order-dependent baseline).
 */
public final class SetLearning {

    /** The set function to learn. */
    public enum Task {
        /** maximum of scalars */
        MAX,
        /** number of distinct tokens */
        UNIQUE,
        /** the two cluster centres */
        CLUSTERS
    }

    /** The compared models. */
    public enum Architecture {
        /** sum-pooled MLPs */
        DEEP_SETS,
        /** ISAB + PMA */
        SET_TRANSFORMER,
        /** MLP on sorted, padded input */
        SORTED_MLP
    }

    /** The largest set; the padded MLP input has this many slots. */
    public static final int MAX_SET = 12;
    private static final int TOKENS = 6;

    private SetLearning() {
    }

    private static int features(Task t) {
        return switch (t) {
            case MAX -> 1;
            case UNIQUE -> TOKENS;
            case CLUSTERS -> 2;
        };
    }

    private static int outputs(Task t) {
        return t == Task.CLUSTERS ? 4 : 1;
    }

    /** One batch: {@code elements [b, n, f]}, targets {@code [b, outputs]}, sort keys per element. */
    private record Batch(float[] elements, float[] targets, int b, int n) {
    }

    private static Batch batch(Task t, int b, int n, SplittableRandom r) {
        int f = features(t);
        float[] x = new float[b * n * f];
        float[] y = new float[b * outputs(t)];
        for (int i = 0; i < b; i++) {
            switch (t) {
                case MAX -> {
                    float m = 0;
                    for (int j = 0; j < n; j++) {
                        float v = (float) r.nextDouble();
                        x[i * n + j] = v;
                        m = Math.max(m, v);
                    }
                    y[i] = m;
                }
                case UNIQUE -> {
                    boolean[] seen = new boolean[TOKENS];
                    int u = 0;
                    for (int j = 0; j < n; j++) {
                        int k = r.nextInt(TOKENS);
                        x[(i * n + j) * f + k] = 1;
                        if (!seen[k]) {
                            seen[k] = true;
                            u++;
                        }
                    }
                    y[i] = u;
                }
                case CLUSTERS -> {
                    float[] c = new float[4];
                    for (int k = 0; k < 4; k++) {
                        c[k] = (float) (4 * r.nextDouble() - 2);
                    }
                    for (int j = 0; j < n; j++) {
                        int k = j < 2 ? j : r.nextInt(2);
                        x[(i * n + j) * 2] = c[2 * k] + (float) (0.3 * r.nextGaussian());
                        x[(i * n + j) * 2 + 1] = c[2 * k + 1] + (float) (0.3 * r.nextGaussian());
                    }
                    System.arraycopy(c, 0, y, 4 * i, 4);
                }
            }
        }
        return new Batch(x, y, b, n);
    }

    /** Sorts each set's elements (by value, token id, or x coordinate) and pads to 12 slots with a flag. */
    private static Tensor padded(Task t, Batch d) {
        int f = features(t);
        int slot = f + 1;
        float[] out = new float[d.b() * MAX_SET * slot];
        for (int i = 0; i < d.b(); i++) {
            Integer[] order = new Integer[d.n()];
            double[] key = new double[d.n()];
            for (int j = 0; j < d.n(); j++) {
                order[j] = j;
                int base = (i * d.n() + j) * f;
                key[j] = switch (t) {
                    case MAX, CLUSTERS -> d.elements()[base];
                    case UNIQUE -> argmax(d.elements(), base, f);
                };
            }
            Arrays.sort(order, (a, c) -> Double.compare(key[c], key[a]));
            for (int j = 0; j < d.n(); j++) {
                int src = (i * d.n() + order[j]) * f;
                int dst = (i * MAX_SET + j) * slot;
                System.arraycopy(d.elements(), src, out, dst, f);
                out[dst + f] = 1;
            }
        }
        return Tensor.of(Shape.of(d.b(), MAX_SET * slot), out);
    }

    private static int argmax(float[] v, int base, int f) {
        int best = 0;
        for (int k = 1; k < f; k++) {
            if (v[base + k] > v[base + best]) {
                best = k;
            }
        }
        return best;
    }

    private static final class Net {
        private final SetEncoder set;
        private final Model mlp;

        Net(Task t, Architecture a, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            if (a == Architecture.SORTED_MLP) {
                set = null;
                mlp = Model.sequential().input(MAX_SET * (features(t) + 1)).seed(seed).dense(64, Activation.RELU)
                        .dense(64, Activation.RELU).dense(outputs(t), Activation.IDENTITY).build();
            } else {
                set = new SetEncoder(features(t), 32, outputs(t), a == Architecture.DEEP_SETS ? SetEncoder.Kind.DEEP_SETS
                        : SetEncoder.Kind.SET_TRANSFORMER, 8, 4, r);
                mlp = null;
            }
        }

        Var forward(Task t, Batch d, boolean training) {
            if (set != null) {
                return set.forward(Var.of(Tensor.of(Shape.of(d.b(), d.n(), features(t)), d.elements())), training);
            }
            return mlp.forward(Var.of(padded(t, d)), training);
        }

        List<Var> parameters() {
            return set != null ? set.parameters() : mlp.parameters();
        }
    }

    private static Var loss(Task t, Var pred, Batch d) {
        Tensor y = Tensor.of(Shape.of(d.b(), outputs(t)), d.targets());
        if (t != Task.CLUSTERS) {
            return pred.sub(Var.of(y)).square().mean();
        }
        Var direct = pred.sub(Var.of(y)).square().sum(1, false);
        Var swapped = pred.sub(Var.of(swap(d))).square().sum(1, false);
        return direct.minimum(swapped).mean();
    }

    private static Tensor swap(Batch d) {
        float[] s = new float[d.targets().length];
        for (int i = 0; i < d.b(); i++) {
            for (int k = 0; k < 2; k++) {
                s[4 * i + k] = d.targets()[4 * i + 2 + k];
                s[4 * i + 2 + k] = d.targets()[4 * i + k];
            }
        }
        return Tensor.of(Shape.of(d.b(), 4), s);
    }

    /**
     * Trains on sets of 3 to 8 elements and evaluates on sets of 8 and 12.
     *
     * @param t the task
     * @param a the model
     * @param steps Adam steps of 32 sets
     * @param seed the seed
     * @return the metric at {@code n = 8} and {@code n = 12}: MSE for MAX, accuracy for UNIQUE,
     *     mean centre distance for CLUSTERS
     */
    public static double[] run(Task t, Architecture a, int steps, long seed) {
        Net net = new Net(t, a, seed);
        Optimizer opt = Optimizer.adam(a == Architecture.SET_TRANSFORMER ? 3e-3f : 5e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Batch d = batch(t, 32, 3 + r.nextInt(6), r);
            loss(t, net.forward(t, d, true), d).backward();
            opt.step(net.parameters());
        }
        SplittableRandom e = new SplittableRandom(seed + 77);
        return new double[] {evaluate(t, net, batch(t, 300, 8, e)), evaluate(t, net, batch(t, 300, MAX_SET, e))};
    }

    private static double evaluate(Task t, Net net, Batch d) {
        float[] p = net.forward(t, d, false).value().toArray();
        float[] y = d.targets();
        double total = 0;
        for (int i = 0; i < d.b(); i++) {
            switch (t) {
                case MAX -> total += (p[i] - y[i]) * (p[i] - y[i]);
                case UNIQUE -> total += Math.round(p[i]) == Math.round(y[i]) ? 1 : 0;
                case CLUSTERS -> {
                    int o = 4 * i;
                    double direct = dist(p, o, y, o) + dist(p, o + 2, y, o + 2);
                    double swapped = dist(p, o, y, o + 2) + dist(p, o + 2, y, o);
                    total += Math.min(direct, swapped) / 2;
                }
            }
        }
        return total / d.b();
    }

    private static double dist(float[] a, int i, float[] b, int j) {
        return Math.hypot(a[i] - b[j], a[i + 1] - b[j + 1]);
    }
}
