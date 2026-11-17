package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Contrastive losses for two towers that embed paired data of different kinds, such as
 * images and their captions, with a <em>learned</em> temperature. Row {@code i} of the
 * first embedding belongs with row {@code i} of the second; every other pairing in the batch
 * is a negative. Rows are normalized, so similarities are cosines.
 *
 * <ul>
 *   <li>{@link #clip}: CLIP's symmetric InfoNCE (Radford et al.,
 *       <a href="https://arxiv.org/abs/2103.00020">2103.00020</a>): a softmax over each row and
 *       each column of the similarity matrix, scaled by a learned {@code exp(s)}. Core
 *       {@code Functional.infoNce} is the same loss with a fixed temperature.</li>
 *   <li>{@link #sigLip}: SigLIP's pairwise sigmoid loss (Zhai et al.,
 *       <a href="https://arxiv.org/abs/2303.15343">2303.15343</a>): every entry of the
 *       similarity matrix is an independent binary decision, positive on the diagonal,
 *       {@code -log sigmoid(z_ij (exp(t) cos_ij + b))}, summed over the batch and divided
 *       by its size. No softmax normalization across the batch, so it works at any batch
 *       size; the learned bias starts strongly negative to offset the many negatives.</li>
 * </ul>
 *
 * <p>The temperature and bias are parameters: hand {@link #parameters()} to the optimizer
 * along with the two towers.
 */
public final class ContrastivePairLoss {
    private final boolean sigmoid;
    private final Var logScale;
    private final Var bias;

    private ContrastivePairLoss(boolean sigmoid, float scale, float bias) {
        this.sigmoid = sigmoid;
        this.logScale = Var.param(Tensor.full(Shape.of(1, 1), (float) Math.log(scale)));
        this.bias = Var.param(Tensor.full(Shape.of(1, 1), bias));
    }

    /**
     * CLIP's softmax loss.
     *
     * @param temperature the initial temperature, 0.07 in the paper
     * @return the loss
     */
    public static ContrastivePairLoss clip(float temperature) {
        return new ContrastivePairLoss(false, 1 / temperature, 0);
    }

    /**
     * SigLIP's sigmoid loss.
     *
     * @param scale the initial logit scale, 10 in the paper
     * @param bias the initial bias, -10 in the paper
     * @return the loss
     */
    public static ContrastivePairLoss sigLip(float scale, float bias) {
        return new ContrastivePairLoss(true, scale, bias);
    }

    /** @return the learned temperature and (for SigLIP) bias */
    public List<Var> parameters() {
        return sigmoid ? List.of(logScale, bias) : List.of(logScale);
    }

    /** @return the current temperature, {@code exp(-s)} */
    public float temperature() {
        return (float) Math.exp(-logScale.value().item());
    }

    /**
     * Similarity logits of all pairings.
     *
     * @param a first-tower embeddings {@code [batch, dim]}
     * @param b second-tower embeddings {@code [count, dim]}
     * @return {@code [batch, count]}
     */
    public Var logits(Var a, Var b) {
        Var cos = SelfSupervisedLosses.normalize(a).matmul(SelfSupervisedLosses.normalize(b).transpose());
        Var scaled = cos.mul(logScale.exp());
        return sigmoid ? scaled.add(bias) : scaled;
    }

    /**
     * The loss of a batch of pairs.
     *
     * @param a first-tower embeddings {@code [batch, dim]}
     * @param b their partners {@code [batch, dim]}
     * @return the loss, a scalar
     */
    public Var compute(Var a, Var b) {
        if (!a.shape().equals(b.shape()) || a.shape().rank() != 2) {
            throw new IllegalArgumentException("need two [batch, dim] embeddings, got " + a.shape() + " and "
                    + b.shape());
        }
        int n = a.shape().dim(0);
        Var logits = logits(a, b);
        if (!sigmoid) {
            Tensor ids = Tensor.arange(n).to(a.value().device());
            return Functional.crossEntropy(logits, ids).add(Functional.crossEntropy(logits.transpose(), ids))
                    .mul(0.5f);
        }
        float[] z = new float[n * n];
        java.util.Arrays.fill(z, -1);
        for (int i = 0; i < n; i++) {
            z[i * n + i] = 1;
        }
        // -log sigmoid(z * logit) = softplus(-z * logit)
        return logits.mul(Tensor.of(Shape.of(n, n), z, a.value().device())).neg().softplus().sum().div(n);
    }
}
