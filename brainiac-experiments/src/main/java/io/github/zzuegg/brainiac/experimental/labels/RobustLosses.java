package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Losses that resist label noise, as core {@link Loss}es over raw logits {@code [batch, classes]}
 * and class-id targets {@code [batch]}. Cross-entropy fits wrong labels because its gradient is
 * largest where the model disagrees with the label; these bound or rebalance that push:
 *
 * <ul>
 *   <li>{@link #gce} (generalised cross-entropy; Zhang and Sabuncu, NeurIPS 2018,
 *       <a href="https://arxiv.org/abs/1805.07836">arXiv:1805.07836</a>): {@code (1 - p_y^q) / q},
 *       cross-entropy as {@code q -> 0} and MAE at {@code q = 1}.</li>
 *   <li>{@link #sce} (symmetric cross-entropy; Wang et al., ICCV 2019,
 *       <a href="https://arxiv.org/abs/1908.06112">arXiv:1908.06112</a>): {@code alpha CE + beta RCE},
 *       the reverse cross-entropy with {@code log 0} clipped to {@code A = -4}, which equals
 *       {@code -A (1 - p_y)}.</li>
 *   <li>{@link #nceRce} (active-passive loss; Ma et al., ICML 2020,
 *       <a href="https://arxiv.org/abs/2006.13554">arXiv:2006.13554</a>): normalised CE
 *       {@code log p_y / sum_k log p_k} (active) plus RCE (passive).</li>
 *   <li>{@link #jointAsymmetric} (JAL; Wang et al., ICCV 2025,
 *       <a href="https://arxiv.org/abs/2507.17692">arXiv:2507.17692</a>): NCE plus the asymmetric
 *       mean squared error {@code AMSE = (1/K) ||a e_y - p||^2} with {@code a >= 1}, which
 *       satisfies the asymmetric condition for large enough {@code a} (the paper uses
 *       {@code alpha = beta = 1, a = 30} on CIFAR-10).</li>
 *   <li>{@link #mae} and the multiclass {@link #focal} loss as baselines.</li>
 * </ul>
 */
public final class RobustLosses {

    private RobustLosses() {
    }

    /** Clipped log of zero in the reverse cross-entropy. */
    public static final float LOG_ZERO = -4f;

    /** @return plain cross-entropy from logits */
    public static Loss crossEntropy() {
        return Loss.crossEntropyFromLogits(0f);
    }

    /**
     * Generalised cross-entropy.
     *
     * @param q in {@code (0, 1]}, 0.7 in the paper
     * @return the loss
     */
    public static Loss gce(float q) {
        return (logits, ids) -> {
            Var py = trueProbability(logits, ids);
            return py.add(1e-7f).pow(q).neg().add(1f).mul(1f / q).mean();
        };
    }

    /**
     * Symmetric cross-entropy.
     *
     * @param alpha the CE weight, 0.1 on CIFAR-10
     * @param beta the RCE weight, 1 on CIFAR-10
     * @return the loss
     */
    public static Loss sce(float alpha, float beta) {
        return (logits, ids) -> {
            Var ce = Loss.crossEntropyFromLogits(0f).compute(logits, ids);
            return ce.mul(alpha).add(rce(logits, ids).mul(beta));
        };
    }

    /**
     * NCE + RCE, the active-passive loss.
     *
     * @param alpha the NCE weight
     * @param beta the RCE weight
     * @return the loss
     */
    public static Loss nceRce(float alpha, float beta) {
        return (logits, ids) -> nce(logits, ids).mul(alpha).add(rce(logits, ids).mul(beta));
    }

    /**
     * The Joint Asymmetric Loss, NCE + AMSE.
     *
     * @param alpha the NCE weight
     * @param beta the AMSE weight
     * @param a the AMSE target scale, at least 1
     * @return the loss
     */
    public static Loss jointAsymmetric(float alpha, float beta, float a) {
        if (a < 1) {
            throw new IllegalArgumentException("a must be at least 1: " + a);
        }
        return (logits, ids) -> nce(logits, ids).mul(alpha).add(amse(logits, ids, a).mul(beta));
    }

    /** @return the mean absolute error between softmax and one-hot, {@code 2 (1 - p_y)} */
    public static Loss mae() {
        return (logits, ids) -> trueProbability(logits, ids).neg().add(1f).mul(2f).mean();
    }

    /**
     * Multiclass focal loss {@code -(1 - p_y)^gamma log p_y} (Lin et al., 2017).
     *
     * @param gamma the focusing power
     * @return the loss
     */
    public static Loss focal(float gamma) {
        return (logits, ids) -> {
            Var logp = SoftTargets.logSoftmax(logits).mul(oneHot(logits, ids)).sum(-1, false);
            return logp.exp().neg().add(1f).maximum(0f).add(1e-7f).pow(gamma).mul(logp).neg().mean();
        };
    }

    /**
     * Normalised cross-entropy, {@code log p_y / sum_k log p_k}, per row averaged.
     *
     * @param logits logits
     * @param ids class ids
     * @return the loss
     */
    public static Var nce(Var logits, Tensor ids) {
        Var logp = SoftTargets.logSoftmax(logits);
        Var num = logp.mul(oneHot(logits, ids)).sum(-1, false);
        Var den = logp.sum(-1, false);
        return num.div(den).mean();
    }

    /**
     * Reverse cross-entropy, {@code -A (1 - p_y)} averaged.
     *
     * @param logits logits
     * @param ids class ids
     * @return the loss
     */
    public static Var rce(Var logits, Tensor ids) {
        return trueProbability(logits, ids).neg().add(1f).mul(-LOG_ZERO).mean();
    }

    /**
     * The asymmetric mean squared error {@code (1/K) ||a e_y - p||^2} averaged over rows.
     *
     * @param logits logits
     * @param ids class ids
     * @param a the target scale
     * @return the loss
     */
    public static Var amse(Var logits, Tensor ids, float a) {
        int k = logits.shape().dim(-1);
        Var p = SoftTargets.softmax(logits);
        return p.sub(oneHot(logits, ids).mul(a)).square().sum(-1, false).mul(1f / k).mean();
    }

    /**
     * @param logits logits
     * @param ids class ids
     * @return {@code p_y} per row
     */
    public static Var trueProbability(Var logits, Tensor ids) {
        return SoftTargets.softmax(logits).mul(oneHot(logits, ids)).sum(-1, false);
    }

    /**
     * @param logits logits {@code [batch, classes]}
     * @param ids {@code [batch]} class ids
     * @return one-hot rows
     */
    public static Tensor oneHot(Var logits, Tensor ids) {
        return Loss.oneHot(ids, logits.shape()).to(logits.dtype());
    }
}
