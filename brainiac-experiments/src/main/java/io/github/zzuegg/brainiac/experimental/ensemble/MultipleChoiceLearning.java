package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Multiple-choice learning: {@code K} hypothesis heads trained so that, per example, only the
 * best head is penalised (winner takes all), which spreads the heads over the modes of an
 * ambiguous target instead of collapsing them onto the (possibly impossible) mean
 * (Guzman-Rivera et al., NeurIPS 2012; Lee et al., NeurIPS 2016,
 * <a href="https://arxiv.org/abs/1606.07839">arXiv:1606.07839</a>):
 *
 * <ul>
 *   <li>{@link Mode#WTA}: only the winner's loss;</li>
 *   <li>{@link Mode#RELAXED_WTA}: the winner gets {@code 1 - ε}, the others {@code ε / (K - 1)},
 *       so no head is left dead (Rupprecht et al., ICCV 2017,
 *       <a href="https://arxiv.org/abs/1612.00197">arXiv:1612.00197</a>);</li>
 *   <li>{@link Mode#EVOLVING_WTA}: the {@code top} best heads share the loss, with {@code top}
 *       annealed from {@code K} to 1 by the caller (Makansi et al. 2019);</li>
 *   <li>{@link #scoringLoss}: resilient MCL's scoring heads (Letzelter et al., NeurIPS 2023,
 *       <a href="https://arxiv.org/abs/2311.01052">arXiv:2311.01052</a>), trained by binary
 *       cross-entropy to predict which head wins, which turns the hypotheses into a weighted
 *       set and, with a kernel, a density ({@link #kernelNll}).</li>
 * </ul>
 * Both regression ({@link #regressionLoss}) and classification ({@link #classificationLoss})
 * heads are supported; {@link Network} is an MLP with {@code K} hypothesis heads and {@code K}
 * score logits.
 */
public final class MultipleChoiceLearning {

    /** How the loss is shared between heads. */
    public enum Mode {
        /** winner takes all */
        WTA,
        /** relaxed winner takes all */
        RELAXED_WTA,
        /** the best {@code top} heads share */
        EVOLVING_WTA
    }

    private MultipleChoiceLearning() {
    }

    /**
     * The per-head weights of a mode, from per-head losses.
     *
     * @param errors {@code [n, K]} per-head losses
     * @param mode the mode
     * @param epsilon the relaxation of {@link Mode#RELAXED_WTA}
     * @param top the number of sharing heads of {@link Mode#EVOLVING_WTA}
     * @return {@code [n, K]} weights, each row summing to one
     */
    public static Tensor weights(Tensor errors, Mode mode, float epsilon, int top) {
        int n = errors.shape().dim(0);
        int k = errors.shape().dim(1);
        float[] e = errors.toArray();
        float[] w = new float[n * k];
        for (int i = 0; i < n; i++) {
            Integer[] order = new Integer[k];
            for (int j = 0; j < k; j++) {
                order[j] = j;
            }
            int row = i;
            java.util.Arrays.sort(order, (a, b) -> Float.compare(e[row * k + a], e[row * k + b]));
            switch (mode) {
                case WTA -> w[i * k + order[0]] = 1;
                case RELAXED_WTA -> {
                    for (int j = 0; j < k; j++) {
                        w[i * k + j] = k == 1 ? 1 : epsilon / (k - 1);
                    }
                    w[i * k + order[0]] = k == 1 ? 1 : 1 - epsilon;
                }
                case EVOLVING_WTA -> {
                    int t = Math.max(1, Math.min(k, top));
                    for (int j = 0; j < t; j++) {
                        w[i * k + order[j]] = 1f / t;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(n, k), w);
    }

    /**
     * The winner of every row.
     *
     * @param errors {@code [n, K]} per-head losses
     * @return {@code [n, K]} one-hot winners
     */
    public static Tensor winners(Tensor errors) {
        return weights(errors, Mode.WTA, 0, 1);
    }

    /**
     * The MCL loss of regression hypotheses.
     *
     * @param hypotheses {@code [n, K, d]}
     * @param target {@code [n, d]}
     * @param mode the mode
     * @param epsilon the relaxation
     * @param top the sharing heads of {@link Mode#EVOLVING_WTA}
     * @return the scalar loss
     */
    public static Var regressionLoss(Var hypotheses, Tensor target, Mode mode, float epsilon, int top) {
        Var e = squaredErrors(hypotheses, target);
        return e.mul(weights(e.value(), mode, epsilon, top)).sum(1, false).mean();
    }

    /**
     * Per-head squared errors.
     *
     * @param hypotheses {@code [n, K, d]}
     * @param target {@code [n, d]}
     * @return {@code [n, K]}
     */
    public static Var squaredErrors(Var hypotheses, Tensor target) {
        int n = hypotheses.shape().dim(0);
        int d = hypotheses.shape().dim(2);
        return hypotheses.sub(target.reshape(n, 1, d)).square().sum(2, false);
    }

    /**
     * The MCL loss of classification heads.
     *
     * @param logits {@code [n, K, classes]}
     * @param ids class ids {@code [n]}
     * @param mode the mode
     * @param epsilon the relaxation
     * @param top the sharing heads
     * @return the scalar loss
     */
    public static Var classificationLoss(Var logits, Tensor ids, Mode mode, float epsilon, int top) {
        int n = logits.shape().dim(0);
        int c = logits.shape().dim(2);
        float[] y = ids.toArray();
        float[] oh = new float[n * c];
        for (int i = 0; i < n; i++) {
            oh[i * c + (int) y[i]] = 1;
        }
        Var ce = logits.logSoftmax(2).mul(Tensor.of(Shape.of(n, 1, c), oh)).sum(2, false).neg();
        return ce.mul(weights(ce.value(), mode, epsilon, top)).sum(1, false).mean();
    }

    /**
     * Resilient MCL's scoring loss: binary cross-entropy of each score logit against "this head
     * wins".
     *
     * @param scoreLogits {@code [n, K]}
     * @param errors {@code [n, K]} per-head losses (only their values are used)
     * @return the scalar loss
     */
    public static Var scoringLoss(Var scoreLogits, Tensor errors) {
        Tensor win = winners(errors);
        return scoreLogits.softplus().sub(scoreLogits.mul(win)).mean();
    }

    /**
     * The oracle (best-of-K) squared error.
     *
     * @param hypotheses {@code [n][K][d]}
     * @param targets {@code [n][d]}
     * @return the mean over rows of the smallest squared distance
     */
    public static double oracleError(double[][][] hypotheses, double[][] targets) {
        double s = 0;
        for (int i = 0; i < targets.length; i++) {
            double best = Double.POSITIVE_INFINITY;
            for (double[] h : hypotheses[i]) {
                double e = 0;
                for (int j = 0; j < h.length; j++) {
                    e += (h[j] - targets[i][j]) * (h[j] - targets[i][j]);
                }
                best = Math.min(best, e);
            }
            s += best;
        }
        return s / targets.length;
    }

    /**
     * The NLL of targets under the isotropic Gaussian mixture centred on the hypotheses with the
     * given weights.
     *
     * @param hypotheses {@code [n][K][d]}
     * @param weights {@code [n][K]} mixture weights (rows summing to one)
     * @param targets {@code [n][d]}
     * @param sigma the kernel width
     * @return the mean NLL
     */
    public static double kernelNll(double[][][] hypotheses, double[][] weights, double[][] targets, double sigma) {
        double s = 0;
        for (int i = 0; i < targets.length; i++) {
            int d = targets[i].length;
            double lse = Double.NEGATIVE_INFINITY;
            double[] terms = new double[hypotheses[i].length];
            for (int k = 0; k < terms.length; k++) {
                double e = 0;
                for (int j = 0; j < d; j++) {
                    double v = hypotheses[i][k][j] - targets[i][j];
                    e += v * v;
                }
                terms[k] = Math.log(Math.max(1e-12, weights[i][k])) - e / (2 * sigma * sigma)
                        - d * Math.log(sigma * Math.sqrt(2 * Math.PI));
                lse = Math.max(lse, terms[k]);
            }
            double acc = 0;
            for (double t : terms) {
                acc += Math.exp(t - lse);
            }
            s -= lse + Math.log(acc);
        }
        return s / targets.length;
    }

    /** An MLP with {@code K} regression hypotheses of dimension {@code d} and {@code K} scores. */
    public static final class Network {

        private final FeatureClassifier net;
        private final int heads;
        private final int dim;

        /**
         * Creates the network.
         *
         * @param inputs input features
         * @param hidden hidden widths
         * @param heads the hypotheses {@code K}
         * @param dim the target dimension {@code d}
         * @param seed the seed
         */
        public Network(int inputs, int[] hidden, int heads, int dim, long seed) {
            this.heads = heads;
            this.dim = dim;
            this.net = FeatureClassifier.mlp(inputs, hidden, heads * (dim + 1), seed);
        }

        /**
         * @param x the inputs {@code [n, inputs]}
         * @param training the training flag
         * @return {@code {hypotheses [n, K, d], score logits [n, K]}}
         */
        public List<Var> forward(Var x, boolean training) {
            int n = x.shape().dim(0);
            Var out = net.logits(x, training);
            Var hyp = out.slice(1, 0, heads * dim).reshape(n, heads, dim);
            Var score = out.slice(1, heads * dim, heads * (dim + 1));
            return List.of(hyp, score);
        }

        /**
         * One training loss: the MCL loss plus the scoring loss.
         *
         * @param x the inputs
         * @param target the targets {@code [n, d]}
         * @param mode the mode
         * @param epsilon the relaxation
         * @param top the sharing heads
         * @param scoreWeight the weight of the scoring loss
         * @return the scalar loss
         */
        public Var loss(Tensor x, Tensor target, Mode mode, float epsilon, int top, float scoreWeight) {
            List<Var> o = forward(Var.of(x), true);
            Var e = squaredErrors(o.get(0), target);
            Var l = e.mul(weights(e.value(), mode, epsilon, top)).sum(1, false).mean();
            return scoreWeight > 0 ? l.add(scoringLoss(o.get(1), e.value()).mul(scoreWeight)) : l;
        }

        /**
         * @param x inputs
         * @return {@code [n][K][d]} hypotheses
         */
        public double[][][] hypotheses(Tensor x) {
            Tensor h = Var.noGrad(() -> forward(Var.of(x), false).get(0).value());
            int n = h.shape().dim(0);
            float[] v = h.toArray();
            double[][][] out = new double[n][heads][dim];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < heads; k++) {
                    for (int j = 0; j < dim; j++) {
                        out[i][k][j] = v[(i * heads + k) * dim + j];
                    }
                }
            }
            return out;
        }

        /**
         * @param x inputs
         * @return {@code [n][K]} winner probabilities from the scoring heads, normalised per row
         */
        public double[][] scores(Tensor x) {
            Tensor s = Var.noGrad(() -> forward(Var.of(x), false).get(1).value());
            int n = s.shape().dim(0);
            float[] v = s.toArray();
            double[][] out = new double[n][heads];
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int k = 0; k < heads; k++) {
                    out[i][k] = 1 / (1 + Math.exp(-v[i * heads + k]));
                    sum += out[i][k];
                }
                for (int k = 0; k < heads; k++) {
                    out[i][k] /= sum;
                }
            }
            return out;
        }

        /** @return all parameters */
        public List<Var> parameters() {
            return net.parameters();
        }

        /** @return the hypotheses per input */
        public int heads() {
            return heads;
        }
    }
}
