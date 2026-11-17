package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The token-wise constrained fine-tuning objective of "deep safety alignment": a
 * replacement for next-token cross-entropy during (possibly adversarial) fine-tuning of an
 * aligned model that keeps the distribution of the first response tokens close to the
 * aligned model's, where safety alignment is concentrated:
 *
 * <pre>L = mean_t (2 / beta_t) softplus(beta_t (log pi_aligned(y_t) - log pi(y_t)))</pre>
 *
 * <p>For small {@code beta_t} this is ordinary cross-entropy (plus a constant); for large
 * {@code beta_t} the sigmoid saturates once the model's log-probability rises above the
 * aligned model's, so the token cannot be pushed much further from the aligned model. The
 * paper uses a large {@code beta} for the first few response tokens and a small one after.
 *
 * <p>Qi et al. 2025, "Safety Alignment Should Be Made More Than Just a Few Tokens Deep",
 * <a href="https://arxiv.org/abs/2406.05946">arXiv:2406.05946</a> (ICLR 2025 outstanding paper).
 */
public final class TokenwiseConstrainedLoss {

    private TokenwiseConstrainedLoss() {
    }

    /**
     * The loss.
     *
     * @param logProbs the model's log-probabilities of the target tokens, any shape with
     *     {@code n} entries
     * @param alignedLogProbs the aligned model's log-probabilities of the same tokens
     * @param beta the constraint strength per token
     * @return the scalar loss
     */
    public static Var loss(Var logProbs, Tensor alignedLogProbs, float[] beta) {
        Shape s = logProbs.shape();
        float[] inv = new float[beta.length];
        for (int i = 0; i < beta.length; i++) {
            inv[i] = 2 / beta[i];
        }
        Tensor b = Tensor.of(s, beta);
        Tensor w = Tensor.of(s, inv);
        return Var.of(alignedLogProbs.reshape(s.toArray())).sub(logProbs).mul(b).softplus().mul(w).mean();
    }

    /**
     * The paper's schedule: {@code early} for the first {@code count} tokens of each response,
     * {@code late} after.
     *
     * @param positions each token's index inside its response
     * @param count how many first tokens are constrained strongly
     * @param early the strong beta
     * @param late the weak beta
     * @return beta per token
     */
    public static float[] schedule(int[] positions, int count, float early, float late) {
        float[] out = new float[positions.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = positions[i] < count ? early : late;
        }
        return out;
    }
}
