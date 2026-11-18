package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.function.DoubleUnaryOperator;

/**
 * Logit adjustment for long-tailed classes. A model trained on long-tailed data predicts the head
 * classes too often; these shift or scale its scores towards the balanced error.
 *
 * <ul>
 *   <li>{@link #posthocAdditive}: subtract {@code tau log pi_y} from the logits after training
 *       (Menon et al., ICLR 2021, <a href="https://arxiv.org/abs/2007.07314">arXiv:2007.07314</a>),
 *       Fisher-consistent for the balanced error at {@code tau = 1}.</li>
 *   <li>{@link #adjustedLoss}: the logit-adjusted loss, cross-entropy of {@code logits + tau log
 *       pi}; at {@code tau = 1} it is the balanced softmax (Ren et al., NeurIPS 2020,
 *       <a href="https://arxiv.org/abs/2007.10740">arXiv:2007.10740</a>), {@link #balancedSoftmax}.</li>
 *   <li>{@link #multiplicative}: multiplicative logit adjustment (Hasegawa and Sato, ICLR 2025,
 *       <a href="https://arxiv.org/abs/2409.17582">arXiv:2409.17582</a>), {@code n_k^-gamma g_k(x)}.
 *       The paper's theory takes {@code g} to be cosine scores against an equal-norm (ETF)
 *       classifier, {@code gamma} near 0.5 and tuned on validation data; with raw logits of
 *       either sign the scaling is not order-preserving, so {@link #cosineScores} provides the
 *       cosine form.</li>
 *   <li>{@link #tune}: picks {@code tau} or {@code gamma} on a balanced validation split.</li>
 * </ul>
 */
public final class LogitAdjustment {

    private LogitAdjustment() {
    }

    /**
     * @param counts training examples per class
     * @return the class priors {@code pi}
     */
    public static double[] priors(int[] counts) {
        double total = 0;
        for (int c : counts) {
            total += c;
        }
        double[] out = new double[counts.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = counts[i] / total;
        }
        return out;
    }

    /**
     * Post-hoc additive adjustment.
     *
     * @param logits {@code [n, classes]}
     * @param priors the training class priors
     * @param tau the strength, 1 for the Bayes-optimal balanced classifier
     * @return {@code logits - tau log pi}
     */
    public static Tensor posthocAdditive(Tensor logits, double[] priors, double tau) {
        return logits.sub(logRow(priors, (float) tau));
    }

    /**
     * The logit-adjusted training loss.
     *
     * @param priors the training class priors
     * @param tau the strength
     * @return cross-entropy of {@code logits + tau log pi} against class ids
     */
    public static Loss adjustedLoss(double[] priors, double tau) {
        Tensor shift = logRow(priors, (float) tau);
        Loss ce = Loss.crossEntropyFromLogits(0f);
        return (logits, ids) -> ce.compute(logits.add(shift), ids);
    }

    /**
     * The balanced softmax, {@link #adjustedLoss} at {@code tau = 1}.
     *
     * @param counts training examples per class
     * @return the loss
     */
    public static Loss balancedSoftmax(int[] counts) {
        return adjustedLoss(priors(counts), 1);
    }

    /**
     * Multiplicative adjustment.
     *
     * @param scores {@code [n, classes]} scores, ideally cosine scores
     * @param counts training examples per class
     * @param gamma the exponent
     * @return {@code scores * n_k^-gamma}
     */
    public static Tensor multiplicative(Tensor scores, int[] counts, double gamma) {
        float[] f = new float[counts.length];
        for (int k = 0; k < f.length; k++) {
            f[k] = (float) Math.pow(counts[k], -gamma);
        }
        return scores.mul(Tensor.of(Shape.of(1, f.length), f));
    }

    /**
     * Cosine scores of features against classifier weight columns.
     *
     * @param features {@code [n, d]}
     * @param weight {@code [d, classes]}, as a core {@code Dense} layer stores it
     * @return {@code cos(f_i, w_k)}, {@code [n, classes]}
     */
    public static Tensor cosineScores(Tensor features, Tensor weight) {
        Tensor fn = features.div(features.square().sum(1, true).add(1e-12f).sqrt());
        Tensor wn = weight.div(weight.square().sum(0, true).add(1e-12f).sqrt());
        return fn.matmul(wn);
    }

    /**
     * Grid search of a scalar on validation data.
     *
     * @param grid the candidate values
     * @param score the validation score of a value, higher is better
     * @return the best value
     */
    public static double tune(double[] grid, DoubleUnaryOperator score) {
        double best = grid[0];
        double bestScore = Double.NEGATIVE_INFINITY;
        for (double g : grid) {
            double s = score.applyAsDouble(g);
            if (s > bestScore) {
                bestScore = s;
                best = g;
            }
        }
        return best;
    }

    private static Tensor logRow(double[] priors, float tau) {
        float[] v = new float[priors.length];
        for (int k = 0; k < v.length; k++) {
            v[k] = (float) (tau * Math.log(Math.max(priors[k], 1e-12)));
        }
        return Tensor.of(Shape.of(1, v.length), v);
    }
}
