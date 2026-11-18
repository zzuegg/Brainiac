package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Test-time augmentation (Krizhevsky et al. 2012) with learned aggregation (Shanmugam et al.,
 * ICCV 2021, <a href="https://arxiv.org/abs/2011.11156">arXiv:2011.11156</a>): predict every
 * test input under a fixed policy of core {@link Augmentation}s and combine the predictions.
 * Naive averaging can hurt when some augmentations are out of distribution for the model (a
 * mirrored digit is another digit), so the learned aggregators fit the combination on
 * validation data instead:
 *
 * <ul>
 *   <li>{@link Aggregation#MEAN}: the mean of the augmented probabilities;</li>
 *   <li>{@link Aggregation#MAX}: the per-class maximum, renormalised;</li>
 *   <li>{@link Aggregation#LEARNED}: logits {@code Σ_a w_a z_a} with one weight per
 *       augmentation;</li>
 *   <li>{@link Aggregation#LEARNED_PER_CLASS}: {@code Σ_a w_{a,c} z_{a,c}}, one weight per
 *       augmentation and class.</li>
 * </ul>
 * Learned weights start uniform and are fitted by Adam on the validation NLL with a ridge
 * towards uniform.
 */
public final class TestTimeAugmentation {

    /** How augmented predictions are combined. */
    public enum Aggregation {
        /** mean probability */
        MEAN,
        /** per-class maximum */
        MAX,
        /** a weight per augmentation */
        LEARNED,
        /** a weight per augmentation and class */
        LEARNED_PER_CLASS
    }

    /** A fitted aggregator. */
    @FunctionalInterface
    public interface Aggregator {

        /**
         * @param logits {@code [augmentation][sample][class]}
         * @return probabilities {@code [sample][class]}
         */
        double[][] apply(double[][][] logits);
    }

    private TestTimeAugmentation() {
    }

    /** @return the identity augmentation */
    public static Augmentation identity() {
        return (f, o, r) -> {
        };
    }

    /**
     * A fixed shift of single-channel images, filling with zeros.
     *
     * @param height the height
     * @param width the width
     * @param dx the horizontal shift
     * @param dy the vertical shift
     * @return the augmentation
     */
    public static Augmentation shift(int height, int width, int dx, int dy) {
        return (f, o, r) -> {
            float[] copy = new float[height * width];
            System.arraycopy(f, o, copy, 0, copy.length);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int sy = y - dy;
                    int sx = x - dx;
                    f[o + y * width + x] = sy >= 0 && sy < height && sx >= 0 && sx < width ? copy[sy * width + sx]
                            : 0f;
                }
            }
        };
    }

    /**
     * A horizontal mirror of single-channel images.
     *
     * @param height the height
     * @param width the width
     * @return the augmentation
     */
    public static Augmentation mirror(int height, int width) {
        return (f, o, r) -> {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width / 2; x++) {
                    int a = o + y * width + x;
                    int b = o + y * width + width - 1 - x;
                    float t = f[a];
                    f[a] = f[b];
                    f[b] = t;
                }
            }
        };
    }

    /**
     * Applies an augmentation to every row of a batch, with a fixed seed so a random
     * augmentation (such as {@link Augmentation#cropAndFlip}) is the same at every call.
     *
     * @param augmentation the augmentation
     * @param x the inputs {@code [n, features]} (or any shape with the samples first)
     * @param seed the seed
     * @return the augmented copy
     */
    public static Tensor apply(Augmentation augmentation, Tensor x, long seed) {
        float[] v = x.toArray();
        int n = x.shape().dim(0);
        int d = v.length / n;
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < n; i++) {
            augmentation.apply(v, i * d, r);
        }
        return Tensor.of(x.shape(), v);
    }

    /**
     * The logits of every augmentation of the policy.
     *
     * @param model maps inputs to logits {@code [n, classes]}
     * @param x the inputs
     * @param policy the augmentations
     * @param seed the seed of random augmentations
     * @return {@code [augmentation][sample][class]}
     */
    public static double[][][] logits(UnaryOperator<Tensor> model, Tensor x, List<Augmentation> policy, long seed) {
        double[][][] out = new double[policy.size()][][];
        for (int a = 0; a < policy.size(); a++) {
            out[a] = EnsembleAudit.rows2(model.apply(apply(policy.get(a), x, seed + a)));
        }
        return out;
    }

    /**
     * Fits an aggregator on validation logits.
     *
     * @param kind the aggregation
     * @param logits validation logits {@code [augmentation][sample][class]}
     * @param labels validation labels
     * @param ridge the ridge towards uniform weights (learned aggregations only)
     * @return the aggregator
     */
    public static Aggregator fit(Aggregation kind, double[][][] logits, int[] labels, double ridge) {
        int a = logits.length;
        int k = logits[0][0].length;
        return switch (kind) {
            case MEAN -> z -> {
                double[][][] p = new double[z.length][][];
                for (int i = 0; i < z.length; i++) {
                    p[i] = softmax(z[i]);
                }
                return EnsembleAudit.mean(p);
            };
            case MAX -> z -> {
                double[][] out = new double[z[0].length][k];
                for (double[][] za : z) {
                    double[][] p = softmax(za);
                    for (int i = 0; i < p.length; i++) {
                        for (int c = 0; c < k; c++) {
                            out[i][c] = Math.max(out[i][c], p[i][c]);
                        }
                    }
                }
                for (double[] row : out) {
                    double s = 0;
                    for (double v : row) {
                        s += v;
                    }
                    for (int c = 0; c < k; c++) {
                        row[c] /= s;
                    }
                }
                return out;
            };
            case LEARNED, LEARNED_PER_CLASS -> {
                double[][] w = learn(logits, labels, kind == Aggregation.LEARNED_PER_CLASS, ridge, 400);
                yield z -> {
                    double[][] out = new double[z[0].length][];
                    for (int i = 0; i < out.length; i++) {
                        double[] s = new double[k];
                        for (int q = 0; q < a; q++) {
                            for (int c = 0; c < k; c++) {
                                s[c] += w[q][c] * z[q][i][c];
                            }
                        }
                        out[i] = softmax1(s);
                    }
                    return out;
                };
            }
        };
    }

    private static double[][] learn(double[][][] z, int[] y, boolean perClass, double ridge, int iterations) {
        int a = z.length;
        int n = y.length;
        int k = z[0][0].length;
        double[][] w = new double[a][k];
        for (double[] row : w) {
            java.util.Arrays.fill(row, 1.0 / a);
        }
        double[][] m1 = new double[a][k];
        double[][] m2 = new double[a][k];
        for (int it = 1; it <= iterations; it++) {
            double[][] g = new double[a][k];
            for (int i = 0; i < n; i++) {
                double[] s = new double[k];
                for (int q = 0; q < a; q++) {
                    for (int c = 0; c < k; c++) {
                        s[c] += w[q][c] * z[q][i][c];
                    }
                }
                double[] p = softmax1(s);
                for (int c = 0; c < k; c++) {
                    double e = (p[c] - (c == y[i] ? 1 : 0)) / n;
                    for (int q = 0; q < a; q++) {
                        g[q][c] += e * z[q][i][c];
                    }
                }
            }
            if (!perClass) {
                for (int q = 0; q < a; q++) {
                    double s = 0;
                    for (int c = 0; c < k; c++) {
                        s += g[q][c];
                    }
                    java.util.Arrays.fill(g[q], s);
                }
            }
            double c1 = 1 - Math.pow(0.9, it);
            double c2 = 1 - Math.pow(0.999, it);
            for (int q = 0; q < a; q++) {
                for (int c = 0; c < k; c++) {
                    double gr = g[q][c] + 2 * ridge * (w[q][c] - 1.0 / a);
                    m1[q][c] = 0.9 * m1[q][c] + 0.1 * gr;
                    m2[q][c] = 0.999 * m2[q][c] + 0.001 * gr * gr;
                    w[q][c] -= 0.02 * (m1[q][c] / c1) / (Math.sqrt(m2[q][c] / c2) + 1e-8);
                }
            }
        }
        return w;
    }

    static double[][] softmax(double[][] z) {
        double[][] out = new double[z.length][];
        for (int i = 0; i < z.length; i++) {
            out[i] = softmax1(z[i]);
        }
        return out;
    }

    static double[] softmax1(double[] z) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) {
            max = Math.max(max, v);
        }
        double[] p = new double[z.length];
        double s = 0;
        for (int i = 0; i < z.length; i++) {
            p[i] = Math.exp(z[i] - max);
            s += p[i];
        }
        for (int i = 0; i < z.length; i++) {
            p[i] /= s;
        }
        return p;
    }

    /**
     * Counts how the aggregate changes the single-pass prediction.
     *
     * @param single the single-pass probabilities
     * @param aggregated the aggregated probabilities
     * @param labels the labels
     * @return {@code {corrections (wrong to right), corruptions (right to wrong)}}
     */
    public static int[] changes(double[][] single, double[][] aggregated, int[] labels) {
        int fix = 0;
        int broke = 0;
        for (int i = 0; i < labels.length; i++) {
            boolean a = argmax(single[i]) == labels[i];
            boolean b = argmax(aggregated[i]) == labels[i];
            fix += !a && b ? 1 : 0;
            broke += a && !b ? 1 : 0;
        }
        return new int[] {fix, broke};
    }

    static int argmax(double[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            b = p[i] > p[b] ? i : b;
        }
        return b;
    }

    /**
     * The standard small-image policy: identity, four one-pixel shifts and the mirror.
     *
     * @param side the image side
     * @return the policy
     */
    public static List<Augmentation> standardPolicy(int side) {
        List<Augmentation> p = new ArrayList<>();
        p.add(identity());
        p.add(shift(side, side, 1, 0));
        p.add(shift(side, side, -1, 0));
        p.add(shift(side, side, 0, 1));
        p.add(shift(side, side, 0, -1));
        p.add(mirror(side, side));
        return p;
    }
}
