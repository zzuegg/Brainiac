package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;

/**
 * Label-shift estimation and correction. Under label shift {@code p(x | y)} stays fixed and
 * only the class priors move from {@code p(y)} to {@code q(y)}, so a classifier is corrected
 * by reweighting its posteriors with {@code w_y = q(y) / p(y)} and renormalising
 * ({@link #correct}).
 *
 * <ul>
 *   <li>{@link #bbse}: black-box shift estimation solves {@code C w = mu}, with {@code C} the
 *       joint confusion matrix {@code p(yhat = i, y = j)} on held-out source data and
 *       {@code mu} the target's predicted-class frequencies (Lipton et al., ICML 2018,
 *       <a href="https://arxiv.org/abs/1802.03916">arXiv:1802.03916</a>); {@link #rlls} adds
 *       a ridge pull of {@code w} towards 1 (Azizzadenesheli et al., ICLR 2019).</li>
 *   <li>{@link #mlls}: EM on the target prior with the classifier's posteriors (Saerens et al.,
 *       Neural Computation 2002), accurate once the posteriors are calibrated; Garg et al.
 *       (NeurIPS 2020, <a href="https://arxiv.org/abs/2003.07554">arXiv:2003.07554</a>) showed
 *       that bias-corrected temperature scaling ({@link Bcts}) is the calibration it needs.</li>
 *   <li>{@link #mapDirichlet}: the MAP version with a Dirichlet prior on the target priors,
 *       {@code q_k = (sum_i r_ik + alpha_k - 1) / (N + sum alpha - K)}, which stabilises small
 *       or very skewed target samples; and {@link Online}, which replaces the E-step over the
 *       whole target set by a stochastic-approximation step per batch,
 *       {@code q <- q + eta (mean posterior of the batch - q)}, so the prior can track a
 *       drifting stream at {@code O(batch K)} cost. Both follow the batch and online FMAPLS
 *       of Hu and Barria (2025, <a href="https://arxiv.org/abs/2511.18615">arXiv:2511.18615</a>)
 *       with a fixed Dirichlet concentration (FMAPLS also estimates it).</li>
 * </ul>
 */
public final class LabelShiftEstimator {

    private LabelShiftEstimator() {
    }

    /**
     * The joint confusion matrix {@code C[i][j] = p(yhat = i, y = j)}.
     *
     * @param predicted predictions on held-out source data
     * @param labels the true labels
     * @param classes the number of classes
     * @return {@code C}
     */
    public static double[][] confusion(int[] predicted, int[] labels, int classes) {
        double[][] c = new double[classes][classes];
        for (int i = 0; i < labels.length; i++) {
            c[predicted[i]][labels[i]] += 1.0 / labels.length;
        }
        return c;
    }

    /**
     * @param predicted predictions on target data
     * @param classes the number of classes
     * @return the predicted-class frequencies {@code mu}
     */
    public static double[] predictedFrequencies(int[] predicted, int classes) {
        double[] mu = new double[classes];
        for (int p : predicted) {
            mu[p] += 1.0 / predicted.length;
        }
        return mu;
    }

    /**
     * Black-box shift estimation.
     *
     * @param confusion {@link #confusion} on held-out source data
     * @param mu {@link #predictedFrequencies} on the target
     * @return the importance weights {@code w}, clipped at 0
     */
    public static double[] bbse(double[][] confusion, double[] mu) {
        return rlls(confusion, mu, 0);
    }

    /**
     * Regularised BBSE: minimises {@code |C (w - 1) - (mu - C 1)|^2 + delta |w - 1|^2}.
     *
     * @param confusion {@code C}
     * @param mu the target's predicted-class frequencies
     * @param delta the ridge, 0 for plain BBSE
     * @return the importance weights, clipped at 0
     */
    public static double[] rlls(double[][] confusion, double[] mu, double delta) {
        int k = mu.length;
        float[] a = new float[k * k];
        float[] b = new float[k];
        for (int i = 0; i < k; i++) {
            double row = 0;
            for (int j = 0; j < k; j++) {
                a[i * k + j] = (float) confusion[i][j];
                row += confusion[i][j];
            }
            b[i] = (float) (mu[i] - row);
        }
        Tensor theta = Linalg.lstsq(Tensor.of(Shape.of(k, k), a), Tensor.of(Shape.of(k, 1), b),
                Math.max(delta, 1e-9));
        double[] t = theta.toDoubleArray();
        double[] w = new double[k];
        for (int i = 0; i < k; i++) {
            w[i] = Math.max(0, 1 + t[i]);
        }
        return w;
    }

    /**
     * Maximum-likelihood label shift by EM.
     *
     * @param probs calibrated source-model posteriors on the target {@code [n][classes]}
     * @param sourcePrior the training class priors
     * @param iterations EM iterations, such as 100
     * @return the estimated target priors
     */
    public static double[] mlls(double[][] probs, double[] sourcePrior, int iterations) {
        double[] alpha = new double[sourcePrior.length];
        java.util.Arrays.fill(alpha, 1);
        return mapDirichlet(probs, sourcePrior, alpha, iterations);
    }

    /**
     * MAP label shift by EM with a Dirichlet prior on the target priors.
     *
     * @param probs calibrated source-model posteriors on the target
     * @param sourcePrior the training class priors
     * @param alpha the Dirichlet concentration per class (all 1 is {@link #mlls})
     * @param iterations EM iterations
     * @return the estimated target priors
     */
    public static double[] mapDirichlet(double[][] probs, double[] sourcePrior, double[] alpha, int iterations) {
        int k = sourcePrior.length;
        int n = probs.length;
        double[] q = sourcePrior.clone();
        double extra = 0;
        for (double a : alpha) {
            extra += a - 1;
        }
        for (int it = 0; it < iterations; it++) {
            double[] sum = expectedCounts(probs, sourcePrior, q);
            double total = 0;
            double[] next = new double[k];
            for (int c = 0; c < k; c++) {
                next[c] = Math.max(1e-8, (sum[c] + alpha[c] - 1) / (n + extra));
                total += next[c];
            }
            double change = 0;
            for (int c = 0; c < k; c++) {
                next[c] /= total;
                change += Math.abs(next[c] - q[c]);
            }
            q = next;
            if (change < 1e-8) {
                break;
            }
        }
        return q;
    }

    /** The E-step: posteriors reweighted by {@code q / p}, summed over the rows. */
    private static double[] expectedCounts(double[][] probs, double[] p, double[] q) {
        int k = p.length;
        double[] sum = new double[k];
        double[] r = new double[k];
        for (double[] row : probs) {
            double z = 0;
            for (int c = 0; c < k; c++) {
                r[c] = row[c] * q[c] / p[c];
                z += r[c];
            }
            for (int c = 0; c < k; c++) {
                sum[c] += r[c] / z;
            }
        }
        return sum;
    }

    /**
     * Importance weights from priors.
     *
     * @param target the target priors
     * @param source the source priors
     * @return {@code q / p}
     */
    public static double[] weights(double[] target, double[] source) {
        double[] w = new double[target.length];
        for (int i = 0; i < w.length; i++) {
            w[i] = target[i] / source[i];
        }
        return w;
    }

    /**
     * Priors from importance weights.
     *
     * @param weights {@code w}
     * @param source the source priors
     * @return {@code w p}, renormalised
     */
    public static double[] priors(double[] weights, double[] source) {
        double[] q = new double[weights.length];
        double s = 0;
        for (int i = 0; i < q.length; i++) {
            q[i] = weights[i] * source[i];
            s += q[i];
        }
        for (int i = 0; i < q.length; i++) {
            q[i] = s > 0 ? q[i] / s : 1.0 / q.length;
        }
        return q;
    }

    /**
     * Corrects posteriors for label shift.
     *
     * @param probs source-model posteriors {@code [n][classes]}
     * @param weights importance weights {@code q / p}
     * @return the reweighted, renormalised posteriors
     */
    public static double[][] correct(double[][] probs, double[] weights) {
        double[][] out = new double[probs.length][];
        for (int i = 0; i < probs.length; i++) {
            double[] r = new double[weights.length];
            double z = 0;
            for (int c = 0; c < r.length; c++) {
                r[c] = probs[i][c] * weights[c];
                z += r[c];
            }
            for (int c = 0; c < r.length; c++) {
                r[c] = z > 0 ? r[c] / z : 1.0 / r.length;
            }
            out[i] = r;
        }
        return out;
    }

    /**
     * Online prior tracking by stochastic approximation of the EM fixed point.
     */
    public static final class Online {

        private final double[] source;
        private final double[] alpha;
        private final double step;
        private double[] q;

        /**
         * Starts at the source priors.
         *
         * @param sourcePrior the training priors
         * @param alpha the Dirichlet concentration (all 1 for none); it pulls the estimate as
         *     {@code alpha - 1} pseudo-counts per batch, normalised by the batch size
         * @param step the step size {@code eta}; constant to track drift
         */
        public Online(double[] sourcePrior, double[] alpha, double step) {
            this.source = sourcePrior.clone();
            this.alpha = alpha.clone();
            this.step = step;
            this.q = sourcePrior.clone();
        }

        /**
         * Updates the estimate with one batch.
         *
         * @param probs calibrated posteriors of the batch
         * @return the new estimate
         */
        public double[] update(double[][] probs) {
            int k = q.length;
            double[] sum = expectedCounts(probs, source, q);
            double extra = 0;
            for (double a : alpha) {
                extra += a - 1;
            }
            double total = 0;
            double[] next = new double[k];
            for (int c = 0; c < k; c++) {
                double target = (sum[c] + alpha[c] - 1) / (probs.length + extra);
                next[c] = Math.max(1e-8, q[c] + step * (target - q[c]));
                total += next[c];
            }
            for (int c = 0; c < k; c++) {
                next[c] /= total;
            }
            q = next;
            return q.clone();
        }

        /** @return the current estimate */
        public double[] priors() {
            return q.clone();
        }
    }

    /**
     * Bias-corrected temperature scaling: {@code softmax(z / T + b)} fitted by NLL on held-out
     * source logits (Alexandari et al., ICML 2020).
     */
    public static final class Bcts {

        private final double temperature;
        private final double[] bias;

        private Bcts(double temperature, double[] bias) {
            this.temperature = temperature;
            this.bias = bias;
        }

        /**
         * Fits the temperature and per-class biases.
         *
         * @param logits held-out source logits {@code [n, classes]}
         * @param labels their labels
         * @param steps Adam steps, such as 300
         * @return the calibrator
         */
        public static Bcts fit(Tensor logits, int[] labels, int steps) {
            int k = logits.shape().dim(1);
            Var logT = Var.param(Tensor.zeros(Shape.of(1)));
            Var b = Var.param(Tensor.zeros(Shape.of(k)));
            float[] ids = new float[labels.length];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = labels[i];
            }
            Tensor y = Tensor.of(Shape.of(ids.length), ids);
            Optimizer opt = Optimizer.adam(0.02f);
            for (int s = 0; s < steps; s++) {
                Var z = Var.of(logits).div(logT.exp()).add(b);
                io.github.zzuegg.brainiac.nn.Functional.crossEntropy(z, y).backward();
                opt.step(List.of(logT, b));
            }
            return new Bcts(Math.exp(logT.value().item()), b.value().toDoubleArray());
        }

        /** @return the fitted temperature */
        public double temperature() {
            return temperature;
        }

        /**
         * @param logits {@code [n, classes]}
         * @return the calibrated probabilities
         */
        public double[][] probabilities(Tensor logits) {
            int k = bias.length;
            double[] v = logits.toDoubleArray();
            int n = v.length / k;
            double[][] out = new double[n][k];
            for (int i = 0; i < n; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < k; c++) {
                    out[i][c] = v[i * k + c] / temperature + bias[c];
                    max = Math.max(max, out[i][c]);
                }
                double s = 0;
                for (int c = 0; c < k; c++) {
                    out[i][c] = Math.exp(out[i][c] - max);
                    s += out[i][c];
                }
                for (int c = 0; c < k; c++) {
                    out[i][c] /= s;
                }
            }
            return out;
        }
    }
}
