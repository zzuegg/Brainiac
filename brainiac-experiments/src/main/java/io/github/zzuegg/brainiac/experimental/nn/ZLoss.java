package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Cross-entropy from <em>logits</em> plus the z-loss {@code weight * mean(logsumexp(logits)^2)}.
 * The extra term pulls the softmax normalizer {@code log Z} towards 0, which keeps output
 * logits from drifting to huge values; that drift is a common cause of loss spikes and
 * float16/bfloat16 overflow in large-model training (PaLM, OLMo, Chameleon). A drop-in
 * {@link Loss} for a model whose last layer outputs raw logits; targets are class ids or
 * one-hot rows.
 *
 * <p>Paper: Chowdhery et al., "PaLM" (2022), <a href="https://arxiv.org/abs/2204.02311">arXiv:2204.02311</a>,
 * section 5; the term goes back to Devlin's "Fast and Robust Neural Network Joint Models"
 * self-normalization (2014).
 */
public final class ZLoss implements Loss {

    private final float weight;

    /**
     * Creates the loss.
     *
     * @param weight the z-loss weight; PaLM uses {@code 1e-4}
     */
    public ZLoss(float weight) {
        this.weight = weight;
    }

    /**
     * The softmax normalizer of each row, computed stably.
     *
     * @param logits {@code [rows, classes]}
     * @return {@code [rows, 1]}
     */
    public static Var logSumExp(Var logits) {
        Tensor max = logits.value().max(-1, true);
        return logits.sub(max).exp().sum(-1, true).log().add(max);
    }

    @Override
    public Var compute(Var logits, Tensor target) {
        Tensor oneHot = Loss.classIds(logits.shape(), target.shape()) ? Loss.oneHot(target, logits.shape()) : target;
        int rows = Math.toIntExact(logits.shape().size() / logits.shape().dim(-1));
        Var ce = Activation.LOG_SOFTMAX.forward(logits).mul(oneHot).sum().mul(-1f / rows);
        if (weight == 0) {
            return ce;
        }
        return ce.add(logSumExp(logits).square().mean().mul(weight));
    }

    @Override
    public String toString() {
        return "ZLoss(" + weight + ")";
    }
}
