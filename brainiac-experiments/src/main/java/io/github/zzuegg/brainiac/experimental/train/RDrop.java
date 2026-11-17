package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.function.UnaryOperator;

/**
 * R-Drop: run every batch through the model twice with different dropout masks, and add to
 * the cross-entropy of both passes a symmetric KL divergence between their predictions. The
 * extra term makes the random sub-networks that dropout samples agree, which removes the
 * train/inference mismatch of dropout and regularizes noticeably on small datasets.
 *
 * <p>Paper: Liang et al., "R-Drop: Regularized Dropout for Neural Networks" (2021),
 * <a href="https://arxiv.org/abs/2106.14448">arXiv:2106.14448</a>.
 */
public final class RDrop {

    private RDrop() {
    }

    /**
     * The R-Drop loss.
     *
     * @param forward the model in training mode (with dropout), returning logits {@code [rows, classes]}
     * @param input the batch
     * @param ids the class ids {@code [rows]}
     * @param alpha the weight of the consistency term; 0.1 to 5 in the paper
     * @return {@code (ce1 + ce2) / 2 + alpha * (KL(p1 || p2) + KL(p2 || p1)) / 2}
     */
    public static Var loss(UnaryOperator<Var> forward, Var input, Tensor ids, float alpha) {
        Var a = forward.apply(input);
        Var b = forward.apply(input);
        Var ce = Functional.crossEntropy(a, ids).add(Functional.crossEntropy(b, ids)).mul(0.5f);
        return ce.add(symmetricKl(a, b).mul(alpha));
    }

    /**
     * The symmetric KL divergence between the softmax distributions of two logit batches,
     * averaged over rows.
     *
     * @param a logits {@code [rows, classes]}
     * @param b logits {@code [rows, classes]}
     * @return {@code (KL(a || b) + KL(b || a)) / 2}
     */
    public static Var symmetricKl(Var a, Var b) {
        Var la = Activation.LOG_SOFTMAX.forward(a);
        Var lb = Activation.LOG_SOFTMAX.forward(b);
        Var diff = la.sub(lb);
        Var kl = la.exp().mul(diff).sub(lb.exp().mul(diff));
        return kl.sum().mul(0.5f / a.shape().dim(0));
    }
}
