package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * Selective language modelling, Rho-1 (Lin et al. 2024, "Not All Tokens Are What You
 * Need", <a href="https://arxiv.org/abs/2404.07965">arXiv:2404.07965</a>, NeurIPS 2024
 * best-paper runner-up): train only on the tokens that are worth learning. A reference model,
 * trained on a small high-quality set, scores every token of the batch; the tokens whose
 * <em>excess loss</em> {@code loss_model - loss_reference} is largest are the learnable ones
 * that the model has not learned yet. Noise tokens (high loss for both models) and tokens
 * already learned (low loss for both) have a small excess and are dropped. The loss is the
 * mean cross-entropy over the kept {@code keepRatio} share of tokens.
 *
 * <p>{@link #perToken} gives the per-row cross-entropy as a {@link Var}, which the core
 * {@code Functional.crossEntropy} only returns as a mean.
 */
public final class SelectiveTokenLoss {

    private SelectiveTokenLoss() {
    }

    /**
     * The cross-entropy of each row.
     *
     * @param logits {@code [rows, classes]}
     * @param ids {@code [rows]} the target ids
     * @return {@code [rows]} the per-row negative log-likelihood
     */
    public static Var perToken(Var logits, Tensor ids) {
        int rows = logits.shape().dim(0);
        int classes = logits.shape().dim(1);
        Tensor target = ids.reshape(rows, 1).equalTo(Tensor.arange(classes).reshape(1, classes));
        return Activation.LOG_SOFTMAX.forward(logits).mul(target).sum(1, false).neg();
    }

    /**
     * The per-row cross-entropy without a graph, for scoring with a reference model.
     *
     * @param logits {@code [rows, classes]}
     * @param ids {@code [rows]}
     * @return the per-row losses
     */
    public static float[] perToken(Tensor logits, Tensor ids) {
        return perToken(Var.of(logits), ids).value().toArray();
    }

    /**
     * The 0/1 mask of the {@code keepRatio} share of rows with the largest excess loss.
     *
     * @param losses the model's per-row losses
     * @param referenceLosses the reference model's per-row losses
     * @param keepRatio the share kept, in {@code (0, 1]}
     * @return {@code [rows]} 1 for the kept rows
     */
    public static Tensor select(float[] losses, float[] referenceLosses, double keepRatio) {
        int n = losses.length;
        int k = Math.max(1, (int) Math.round(keepRatio * n));
        float[] excess = new float[n];
        for (int i = 0; i < n; i++) {
            excess[i] = losses[i] - referenceLosses[i];
        }
        float[] sorted = excess.clone();
        Arrays.sort(sorted);
        float threshold = sorted[n - k];
        float[] mask = new float[n];
        int kept = 0;
        for (int i = 0; i < n && kept < k; i++) {
            if (excess[i] > threshold) {
                mask[i] = 1;
                kept++;
            }
        }
        for (int i = 0; i < n && kept < k; i++) {
            if (excess[i] == threshold) {
                mask[i] = 1;
                kept++;
            }
        }
        return Tensor.of(Shape.of(n), mask);
    }

    /**
     * Rho-1's loss: the mean cross-entropy over the rows with the largest excess loss.
     *
     * @param logits the model's logits {@code [rows, classes]}
     * @param ids the target ids {@code [rows]}
     * @param referenceLosses the reference model's per-row losses
     * @param keepRatio the share of rows trained on, for example 0.6
     * @return a scalar
     */
    public static Var loss(Var logits, Tensor ids, float[] referenceLosses, double keepRatio) {
        Var nll = perToken(logits, ids);
        Tensor mask = select(nll.value().toArray(), referenceLosses, keepRatio);
        return nll.mul(mask).sum().div(Math.max(1f, mask.sum().item()));
    }
}
