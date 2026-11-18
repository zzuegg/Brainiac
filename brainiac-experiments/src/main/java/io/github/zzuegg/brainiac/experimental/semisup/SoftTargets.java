package io.github.zzuegg.brainiac.experimental.semisup;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Small helpers shared by the semi-supervised and imperfect-label blocks: log-softmax and
 * softmax on {@code Var} logits, soft and per-row weighted cross-entropy, sharpening and
 * conversion between {@code float[][]} probability rows and tensors. Probabilities that serve
 * as targets are plain tensors, so no gradient flows into them (the stop-gradient every
 * consistency method uses).
 */
public final class SoftTargets {

    private SoftTargets() {
    }

    /**
     * @param logits {@code [batch, classes]} logits
     * @return their log-softmax over the last axis
     */
    public static Var logSoftmax(Var logits) {
        return logits.sub(logits.logsumexp(-1, true));
    }

    /**
     * @param logits {@code [batch, classes]} logits
     * @return their softmax over the last axis
     */
    public static Var softmax(Var logits) {
        return logSoftmax(logits).exp();
    }

    /**
     * Softmax rows of a logits tensor as plain arrays.
     *
     * @param logits {@code [batch, classes]}
     * @param temperature divides the logits first, 1 for none
     * @return the probability rows
     */
    public static float[][] probabilities(Tensor logits, float temperature) {
        int classes = logits.shape().dim(-1);
        int rows = logits.shape().dim(0);
        float[] v = logits.toArray();
        float[][] out = new float[rows][classes];
        for (int i = 0; i < rows; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                max = Math.max(max, v[i * classes + c] / temperature);
            }
            double sum = 0;
            for (int c = 0; c < classes; c++) {
                double e = Math.exp(v[i * classes + c] / temperature - max);
                out[i][c] = (float) e;
                sum += e;
            }
            for (int c = 0; c < classes; c++) {
                out[i][c] /= (float) sum;
            }
        }
        return out;
    }

    /**
     * @param rows probability rows
     * @return them as a {@code [rows, classes]} tensor
     */
    public static Tensor tensor(float[][] rows) {
        int c = rows[0].length;
        float[] v = new float[rows.length * c];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, v, i * c, c);
        }
        return Tensor.of(Shape.of(rows.length, c), v);
    }

    /**
     * @param ids class ids
     * @param classes the number of classes
     * @return one-hot rows
     */
    public static float[][] oneHot(int[] ids, int classes) {
        float[][] out = new float[ids.length][classes];
        for (int i = 0; i < ids.length; i++) {
            out[i][ids[i]] = 1;
        }
        return out;
    }

    /**
     * @param ids class ids
     * @return them as a {@code [n]} float tensor, as {@code Functional.crossEntropy} takes
     */
    public static Tensor ids(int[] ids) {
        float[] v = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            v[i] = ids[i];
        }
        return Tensor.of(Shape.of(ids.length), v);
    }

    /**
     * @param rows probability rows
     * @return the argmax of every row
     */
    public static int[] argmax(float[][] rows) {
        int[] out = new int[rows.length];
        for (int i = 0; i < rows.length; i++) {
            int best = 0;
            for (int c = 1; c < rows[i].length; c++) {
                if (rows[i][c] > rows[i][best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }

    /**
     * Temperature sharpening, {@code p_c^(1/T) / sum_k p_k^(1/T)} (MixMatch's entropy
     * minimisation).
     *
     * @param p probability rows
     * @param temperature {@code T}, below 1 to sharpen
     * @return sharpened rows
     */
    public static float[][] sharpen(float[][] p, double temperature) {
        float[][] out = new float[p.length][];
        for (int i = 0; i < p.length; i++) {
            out[i] = new float[p[i].length];
            double sum = 0;
            for (int c = 0; c < p[i].length; c++) {
                double v = Math.pow(Math.max(p[i][c], 1e-12), 1 / temperature);
                out[i][c] = (float) v;
                sum += v;
            }
            for (int c = 0; c < p[i].length; c++) {
                out[i][c] /= (float) sum;
            }
        }
        return out;
    }

    /**
     * Cross-entropy against soft target rows, each row weighted.
     *
     * @param logits {@code [batch, classes]} logits
     * @param targets {@code [batch, classes]} target probabilities
     * @param weights one weight per row, or null for all ones
     * @param denominator what the weighted sum is divided by (the batch size for FixMatch's
     *     masked mean)
     * @return {@code -sum_i w_i sum_c t_ic log p_ic / denominator}
     */
    public static Var softCrossEntropy(Var logits, Tensor targets, float[] weights, float denominator) {
        Var perRow = logSoftmax(logits).mul(targets).sum(-1, false).neg();
        if (weights != null) {
            perRow = perRow.mul(Tensor.of(Shape.of(weights.length), weights));
        }
        return perRow.sum().mul(1f / denominator);
    }

    /**
     * The mean entropy of the predicted distributions, {@code -mean_i sum_c p log p}: the
     * entropy-minimisation term of VAT and pseudo-labelling.
     *
     * @param logits {@code [batch, classes]} logits
     * @return the mean entropy, differentiable
     */
    public static Var entropy(Var logits) {
        Var logp = logSoftmax(logits);
        return logp.exp().mul(logp).sum(-1, false).neg().mean();
    }

    /**
     * {@code KL(target || softmax(logits))} averaged over rows.
     *
     * @param target {@code [batch, classes]} target probabilities, constant
     * @param logits {@code [batch, classes]} logits
     * @return the mean KL divergence
     */
    public static Var kl(Tensor target, Var logits) {
        Tensor tLogT = target.mul(target.add(1e-8f).log()).sum(-1, false);
        Var cross = logSoftmax(logits).mul(target).sum(-1, false);
        return Var.of(tLogT).sub(cross).mean();
    }

    /**
     * Selects rows of a tensor.
     *
     * @param t the tensor, rows on axis 0
     * @param rows the row indices
     * @return those rows
     */
    public static Tensor rows(Tensor t, int[] rows) {
        float[] idx = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            idx[i] = rows[i];
        }
        return t.gather(0, Tensor.of(Shape.of(rows.length), idx));
    }
}
