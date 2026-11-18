package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Partial-label learning: every training example comes with a set of candidate labels, one of
 * them correct.
 *
 * <ul>
 *   <li>PRODEN (Lv et al., ICML 2020, <a href="https://arxiv.org/abs/2002.08053">arXiv:2002.08053</a>):
 *       a per-example weight vector over the candidates, uniform at first; the loss is the
 *       weighted cross-entropy ({@link #prodenLoss}), and the weights are re-set to the model's
 *       predictions restricted to the candidate set and renormalised ({@link #update}), so the
 *       model progressively identifies the true label.</li>
 *   <li>{@link #uniformLoss}: the same loss with the weights kept uniform (no re-weighting), the
 *       baseline.</li>
 *   <li>{@link #ccLoss}: the classifier-consistent loss {@code -log sum_{c in S} p_c} (Feng et al.,
 *       NeurIPS 2020), the likelihood that the prediction lands in the candidate set.</li>
 * </ul>
 */
public final class PartialLabels {

    private final boolean[][] candidates;
    private final float[][] weights;

    /**
     * Creates the weights, uniform over each candidate set.
     *
     * @param candidates {@code [examples][classes]} candidate masks
     */
    public PartialLabels(boolean[][] candidates) {
        this.candidates = candidates;
        this.weights = new float[candidates.length][];
        for (int i = 0; i < candidates.length; i++) {
            weights[i] = uniform(candidates[i]);
        }
    }

    private static float[] uniform(boolean[] mask) {
        int k = 0;
        for (boolean b : mask) {
            k += b ? 1 : 0;
        }
        float[] w = new float[mask.length];
        for (int c = 0; c < mask.length; c++) {
            w[c] = mask[c] ? 1f / k : 0;
        }
        return w;
    }

    /**
     * PRODEN's weighted cross-entropy.
     *
     * @param logits the batch's logits
     * @param indices the batch's example indices
     * @return {@code -mean_i sum_c w_ic log p_ic}
     */
    public Var prodenLoss(Var logits, int[] indices) {
        float[][] w = new float[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            w[i] = weights[indices[i]];
        }
        return SoftTargets.softCrossEntropy(logits, SoftTargets.tensor(w), null, indices.length);
    }

    /**
     * PRODEN's weight update from the current predictions (the paper updates after each
     * minibatch with the new predictions).
     *
     * @param logits the batch's logits (values only)
     * @param indices the batch's example indices
     */
    public void update(Tensor logits, int[] indices) {
        float[][] p = SoftTargets.probabilities(logits, 1f);
        for (int i = 0; i < indices.length; i++) {
            boolean[] mask = candidates[indices[i]];
            double s = 0;
            for (int c = 0; c < mask.length; c++) {
                s += mask[c] ? p[i][c] : 0;
            }
            for (int c = 0; c < mask.length; c++) {
                weights[indices[i]][c] = mask[c] ? (float) (p[i][c] / Math.max(s, 1e-12)) : 0;
            }
        }
    }

    /**
     * The uniform-candidate loss.
     *
     * @param logits the batch's logits
     * @param indices the batch's example indices
     * @param candidates the candidate masks
     * @return {@code -mean_i (1/|S_i|) sum_{c in S_i} log p_ic}
     */
    public static Var uniformLoss(Var logits, int[] indices, boolean[][] candidates) {
        float[][] w = new float[indices.length][];
        for (int i = 0; i < indices.length; i++) {
            w[i] = uniform(candidates[indices[i]]);
        }
        return SoftTargets.softCrossEntropy(logits, SoftTargets.tensor(w), null, indices.length);
    }

    /**
     * The classifier-consistent loss.
     *
     * @param logits the batch's logits
     * @param indices the batch's example indices
     * @param candidates the candidate masks
     * @return {@code -mean_i log sum_{c in S_i} p_ic}
     */
    public static Var ccLoss(Var logits, int[] indices, boolean[][] candidates) {
        int classes = logits.shape().dim(-1);
        float[] mask = new float[indices.length * classes];
        for (int i = 0; i < indices.length; i++) {
            for (int c = 0; c < classes; c++) {
                mask[i * classes + c] = candidates[indices[i]][c] ? 1 : 0;
            }
        }
        Var p = SoftTargets.softmax(logits).mul(Tensor.of(Shape.of(indices.length, classes), mask));
        return p.sum(-1, false).add(1e-7f).log().neg().mean();
    }

    /**
     * @param index an example
     * @return its current candidate weights; a copy
     */
    public float[] weights(int index) {
        return weights[index].clone();
    }
}
