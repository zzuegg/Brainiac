package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Preference and alignment losses for language models, on sequence log-probabilities
 * {@code log pi(y | x)} ({@code [batch]} values, summed or averaged over the response tokens):
 * <ul>
 * <li>{@link #bradleyTerry}: reward-model training (Christiano et al. 2017; Ouyang et al.
 *     2022, <a href="https://arxiv.org/abs/2203.02155">arXiv:2203.02155</a>),
 *     {@code -log sigmoid(r_chosen - r_rejected)};</li>
 * <li>{@link #ipo}: identity preference optimization (Azar et al. 2023,
 *     <a href="https://arxiv.org/abs/2310.12036">arXiv:2310.12036</a>), a squared loss that
 *     pulls the log-ratio margin to {@code 1 / (2 tau)} instead of pushing it to infinity, so
 *     it does not overfit deterministic preferences as DPO does;</li>
 * <li>{@link #kto}: Kahneman-Tversky optimization (Ethayarajh et al. 2024,
 *     <a href="https://arxiv.org/abs/2402.01306">arXiv:2402.01306</a>), which needs only a
 *     thumbs-up or thumbs-down per answer, not pairs: a prospect-theory value of the reward
 *     {@code beta log(pi / pi_ref)} against a reference point {@code z} (the policy's KL from
 *     the reference, estimated on mismatched pairs, see {@link #klEstimate});</li>
 * <li>{@link #orpo}: odds-ratio preference optimization (Hong et al. 2024,
 *     <a href="https://arxiv.org/abs/2403.07691">arXiv:2403.07691</a>), supervised fine-tuning
 *     plus {@code -lambda log sigmoid(log odds(chosen) - log odds(rejected))} on the mean token
 *     probabilities, with no reference model;</li>
 * <li>{@link #simpo}: simple preference optimization (Meng et al. 2024,
 *     <a href="https://arxiv.org/abs/2405.14734">arXiv:2405.14734</a>),
 *     {@code -log sigmoid(beta (mean log pi(y_w) - mean log pi(y_l)) - gamma)}: length-normalized,
 *     reference-free, with a target margin {@code gamma}.</li>
 * </ul>
 * All return the mean over the batch. DPO itself is in the reasoning experiments.
 */
public final class PreferenceLosses {

    private PreferenceLosses() {
    }

    /**
     * @param x any values
     * @return {@code log sigmoid(x)}, computed as {@code -softplus(-x)}
     */
    public static Var logSigmoid(Var x) {
        return x.neg().softplus().neg();
    }

    /**
     * @param chosen rewards of the preferred answers {@code [batch]}
     * @param rejected rewards of the other answers
     * @return the Bradley-Terry negative log-likelihood
     */
    public static Var bradleyTerry(Var chosen, Var rejected) {
        return logSigmoid(chosen.sub(rejected)).mean().neg();
    }

    /**
     * @param policyChosen {@code log pi(y_w | x)}
     * @param policyRejected {@code log pi(y_l | x)}
     * @param refChosen {@code log pi_ref(y_w | x)}
     * @param refRejected {@code log pi_ref(y_l | x)}
     * @param tau the regularization; the target margin is {@code 1 / (2 tau)}
     * @return the IPO loss
     */
    public static Var ipo(Var policyChosen, Var policyRejected, Var refChosen, Var refRejected, float tau) {
        Var h = policyChosen.sub(refChosen).sub(policyRejected.sub(refRejected));
        return h.sub(1 / (2 * tau)).square().mean();
    }

    /**
     * @param policy {@code log pi(y | x)} per example
     * @param reference {@code log pi_ref(y | x)}
     * @param desirable whether each answer was good
     * @param beta the reward scale, for example 0.1
     * @param lambdaDesirable weight of good answers, for example 1
     * @param lambdaUndesirable weight of bad answers, for example 1
     * @param klReference the reference point {@code z}, from {@link #klEstimate} (a constant)
     * @return the KTO loss
     */
    public static Var kto(Var policy, Var reference, boolean[] desirable, float beta, float lambdaDesirable,
            float lambdaUndesirable, float klReference) {
        int n = desirable.length;
        float[] sign = new float[n];
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) {
            sign[i] = desirable[i] ? 1 : -1;
            weight[i] = desirable[i] ? lambdaDesirable : lambdaUndesirable;
        }
        Tensor s = Tensor.of(Shape.of(n), sign);
        Var r = policy.sub(reference);
        Var v = r.sub(klReference).mul(s).mul(beta).sigmoid();
        return v.neg().add(1).mul(Tensor.of(Shape.of(n), weight)).mean();
    }

    /**
     * The KTO reference point: the policy's KL divergence from the reference, estimated as
     * the mean log-ratio on mismatched (prompt, answer) pairs, clamped at zero, without
     * gradient.
     *
     * @param policyMismatched {@code log pi(y' | x)} for answers of other prompts
     * @param referenceMismatched {@code log pi_ref(y' | x)}
     * @return the estimate
     */
    public static float klEstimate(Var policyMismatched, Var referenceMismatched) {
        return Math.max(0, policyMismatched.value().sub(referenceMismatched.value()).mean().item());
    }

    /**
     * @param nllChosen the mean token negative log-likelihood of the preferred answers (the SFT term)
     * @param meanLogpChosen the mean token log-probability of each preferred answer {@code [batch]}
     * @param meanLogpRejected the same for the other answers
     * @param lambda the weight of the odds-ratio term, for example 0.1 to 1
     * @return the ORPO loss
     */
    public static Var orpo(Var nllChosen, Var meanLogpChosen, Var meanLogpRejected, float lambda) {
        Var ratio = logOdds(meanLogpChosen).sub(logOdds(meanLogpRejected));
        return nllChosen.sub(logSigmoid(ratio).mean().mul(lambda));
    }

    /** {@code log(p / (1 - p))} from {@code log p}. */
    private static Var logOdds(Var logp) {
        return logp.sub(logp.exp().minimum(1 - 1e-6f).neg().add(1).log());
    }

    /**
     * @param meanLogpChosen the mean token log-probability of each preferred answer
     * @param meanLogpRejected the same for the other answers
     * @param beta the scale, for example 2
     * @param gamma the target margin, for example 0.5
     * @return the SimPO loss
     */
    public static Var simpo(Var meanLogpChosen, Var meanLogpRejected, float beta, float gamma) {
        return logSigmoid(meanLogpChosen.sub(meanLogpRejected).mul(beta).sub(gamma)).mean().neg();
    }
}
