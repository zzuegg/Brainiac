package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Ordinal-regression heads for any network: a {@link Layer} that maps features {@code [n, h]}
 * to {@code K - 1} cumulative logits {@code logit P(y > k)}, with its matching loss and
 * decoding, for ordered labels {@code 0 .. K-1} (ratings, severity grades) where softmax ignores
 * the order and regression assumes equal spacing.
 *
 * <ul>
 *   <li>{@link Kind#PROPORTIONAL_ODDS} (McCullagh, JRSS-B 1980): one score {@code s = w.h} and
 *       ordered cut-points {@code theta_k} (a first value plus softplus increments), so
 *       {@code P(y > k) = sigmoid(s - theta_k)}; trained by the categorical likelihood.</li>
 *   <li>{@link Kind#CORAL} (Cao, Mirjalili and Raschka, PRL 2020,
 *       <a href="https://arxiv.org/abs/1901.07884">arXiv:1901.07884</a>): {@code K - 1} binary
 *       tasks "y &gt; k" sharing the weight vector with separate biases, trained by summed binary
 *       cross-entropy; rank-consistent when the biases are ordered (they are at the optimum).</li>
 *   <li>{@link Kind#CORN} (Shi, Cao and Raschka, 2023,
 *       <a href="https://arxiv.org/abs/2111.08851">arXiv:2111.08851</a>): unshared logits, task
 *       {@code k} trained only on examples with {@code y >= k} to estimate the conditional
 *       {@code P(y > k | y >= k)}; {@code P(y > k)} is the running product, which is
 *       rank-consistent by construction.</li>
 * </ul>
 * Prediction counts the cumulative probabilities above one half.
 */
public final class OrdinalRegression implements Layer {

    /** The head type. */
    public enum Kind {
        /** Proportional odds (cumulative logit). */
        PROPORTIONAL_ODDS,
        /** CORAL. */
        CORAL,
        /** CORN. */
        CORN
    }

    private final Kind kind;
    private final int classes;
    private final Var weight;
    private final Var bias;

    /**
     * Creates a head.
     *
     * @param kind the head type
     * @param inputSize features per row
     * @param classes the number of ordered classes {@code K}
     * @param seed the initialisation seed
     */
    public OrdinalRegression(Kind kind, int inputSize, int classes, long seed) {
        this.kind = kind;
        this.classes = classes;
        int outs = kind == Kind.CORN ? classes - 1 : 1;
        float bound = (float) Math.sqrt(6.0 / (inputSize + outs));
        this.weight = Var.param(Tensor.uniform(Shape.of(inputSize, outs), -bound, bound, new SplittableRandom(seed)));
        float[] b = new float[classes - 1];
        for (int k = 0; k < classes - 1; k++) {
            // spread initial thresholds (PO stores the first cut-point then log-increments)
            b[k] = kind == Kind.PROPORTIONAL_ODDS ? (k == 0 ? -1f : 0f) : 1f - 2f * k / Math.max(1, classes - 2);
        }
        this.bias = Var.param(Tensor.of(Shape.of(1, classes - 1), b));
    }

    /**
     * The cumulative logits {@code logit P(y > k)} (for CORN, the conditional logits).
     *
     * @param input features {@code [n, h]}
     * @return logits {@code [n, K - 1]}
     */
    @Override
    public Var forward(Var input) {
        Var s = input.matmul(weight);
        return switch (kind) {
            case CORAL -> s.add(bias);
            case CORN -> s.add(bias);
            case PROPORTIONAL_ODDS -> s.sub(cutPoints());
        };
    }

    private Var cutPoints() {
        // theta_0 = b_0, theta_k = theta_{k-1} + softplus(b_k)
        Var first = bias.slice(1, 0, 1);
        if (classes == 2) {
            return first;
        }
        Var inc = bias.slice(1, 1, classes - 1).softplus();
        return Var.concat(1, first, first.add(inc.cumsum(1)));
    }

    /**
     * The training loss for the head's outputs.
     *
     * @param logits the head output {@code [n, K - 1]}
     * @param labels the ordered labels in {@code 0 .. K-1}
     * @return the mean loss
     */
    public Var loss(Var logits, int[] labels) {
        int n = labels.length;
        int k1 = classes - 1;
        float[] target = new float[n * k1];
        float[] mask = new float[n * k1];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < k1; k++) {
                target[i * k1 + k] = labels[i] > k ? 1 : 0;
                mask[i * k1 + k] = kind != Kind.CORN || labels[i] >= k ? 1 : 0;
            }
        }
        Tensor t = Tensor.of(Shape.of(n, k1), target);
        if (kind == Kind.PROPORTIONAL_ODDS) {
            Var cum = logits.sigmoid();
            Var ones = Var.of(Tensor.of(Shape.of(n, 1), filled(n, 1f)));
            Var zeros = Var.of(Tensor.zeros(Shape.of(n, 1)));
            Var padded = Var.concat(1, ones, cum, zeros);
            Var probs = padded.slice(1, 0, classes).sub(padded.slice(1, 1, classes + 1));
            float[] onehot = new float[n * classes];
            for (int i = 0; i < n; i++) {
                onehot[i * classes + labels[i]] = 1;
            }
            Var picked = probs.mul(Tensor.of(Shape.of(n, classes), onehot)).sum(1, false);
            return picked.add(1e-6f).log().mean().neg();
        }
        // binary cross-entropy with logits: softplus(z) - t z, masked
        Var bce = logits.softplus().sub(logits.mul(t)).mul(Tensor.of(Shape.of(n, k1), mask));
        float count = 0;
        for (float m : mask) {
            count += m;
        }
        return bce.sum().div(Math.max(1f, count / k1));
    }

    /**
     * The binary-relevance loss of {@code K - 1} independent "y &gt; k" tasks on any logits (the
     * unshared, unconditional baseline that CORAL and CORN fix).
     *
     * @param logits {@code [n, K - 1]}
     * @param labels the labels
     * @return the mean loss
     */
    public static Var independentTasksLoss(Var logits, int[] labels) {
        int n = labels.length;
        int k1 = logits.shape().dim(1);
        float[] target = new float[n * k1];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < k1; k++) {
                target[i * k1 + k] = labels[i] > k ? 1 : 0;
            }
        }
        Var t = Var.of(Tensor.of(Shape.of(n, k1), target));
        return logits.softplus().sub(logits.mul(t)).sum().div(n);
    }

    /**
     * The cumulative probabilities {@code P(y > k)} from the head output.
     *
     * @param logits {@code [n, K - 1]}
     * @return {@code [n][K - 1]}
     */
    public double[][] cumulative(Tensor logits) {
        int n = logits.shape().dim(0);
        int k1 = classes - 1;
        float[] z = logits.toArray();
        double[][] p = new double[n][k1];
        for (int i = 0; i < n; i++) {
            double run = 1;
            for (int k = 0; k < k1; k++) {
                double s = 1 / (1 + Math.exp(-z[i * k1 + k]));
                run = kind == Kind.CORN ? run * s : s;
                p[i][k] = run;
            }
        }
        return p;
    }

    /**
     * Decodes labels: the number of cumulative probabilities above one half.
     *
     * @param cumulative {@code P(y > k)} per row
     * @return the predicted labels
     */
    public static int[] decode(double[][] cumulative) {
        int[] out = new int[cumulative.length];
        for (int i = 0; i < out.length; i++) {
            for (double p : cumulative[i]) {
                if (p > 0.5) {
                    out[i]++;
                }
            }
        }
        return out;
    }

    /**
     * The share of rows whose cumulative probabilities increase somewhere (rank inconsistency).
     *
     * @param cumulative {@code P(y > k)} per row
     * @return the violation rate
     */
    public static double violations(double[][] cumulative) {
        int v = 0;
        for (double[] row : cumulative) {
            for (int k = 1; k < row.length; k++) {
                if (row[k] > row[k - 1] + 1e-9) {
                    v++;
                    break;
                }
            }
        }
        return (double) v / cumulative.length;
    }

    private static float[] filled(int n, float v) {
        float[] a = new float[n];
        java.util.Arrays.fill(a, v);
        return a;
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return classes - 1;
    }

    /**
     * The head type.
     *
     * @return the kind
     */
    public Kind kind() {
        return kind;
    }
}
