package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * DPO with other link functions: every DPO-family loss here is
 * {@code -log sigmoid(beta (psi(h_w) - psi(h_l)))} with {@code h = log pi(y) - log ref(y)} the
 * log-ratio of a response and {@code psi} the derivative {@code f'(t)} of the f-divergence
 * that regularizes the underlying RLHF objective, evaluated at {@code t = pi / ref = exp(h)}
 * (f-DPO, Wang et al. 2023, <a href="https://arxiv.org/abs/2309.16240">arXiv:2309.16240</a>):
 * <ul>
 *   <li>{@link Link#REVERSE_KL}: {@code f(t) = t log t}, {@code psi(h) = h}: plain DPO;</li>
 *   <li>{@link Link#CHI_SQUARED}: chi-squared preference optimization (chiPO, Huang et al.,
 *       ICLR 2025, <a href="https://arxiv.org/abs/2407.13399">arXiv:2407.13399</a>),
 *       {@code f(t) = (t - 1)^2 / 2 + t log t}, {@code psi(h) = exp(h) + h}, i.e.
 *       {@code phi(z) = z + log z}. The chi-squared term penalizes leaving the reference's
 *       support far more than KL does, which makes the method provably robust to
 *       over-optimization; the paper also clips the margin to {@code [-2 R, 2 R]};</li>
 *   <li>{@link Link#SQUARED_LOG}: SquaredPO (ICLR 2026,
 *       <a href="https://arxiv.org/abs/2602.06788">arXiv:2602.06788</a>), the non-convex
 *       {@code f(t) = (log t)^2 / 2} with its minimum at {@code t = 1}, {@code psi(h) = h exp(-h)}.
 *       Its gradient is DPO's with an effective {@code beta / t}: when the chosen response's
 *       likelihood falls below the reference ({@code t < 1}) the regularization strengthens
 *       and pushes it back up, against likelihood displacement;</li>
 *   <li>{@link Link#JENSEN_SHANNON}: {@code psi(h) = log(2 t / (1 + t))}.</li>
 * </ul>
 * {@link #fDpo} takes any {@code psi}. {@link #batch} computes the four sequence
 * log-likelihoods of a batch of preferences on a {@link WindowLm} and its reference.
 */
public final class LinkDpoLosses {

    /** The built-in links. */
    public enum Link {
        /** Plain DPO. */
        REVERSE_KL,
        /** chiPO: {@code exp(h) + h}. */
        CHI_SQUARED,
        /** SquaredPO: {@code h exp(-h)}. */
        SQUARED_LOG,
        /** Jensen-Shannon: {@code log(2 t / (1 + t))}. */
        JENSEN_SHANNON
    }

    /** The largest log-ratio fed to an exponential link, against overflow. */
    public static final float MAX_LOG_RATIO = 10;

    /**
     * Sequence log-likelihoods of a batch of preferences.
     *
     * @param policyChosen {@code log pi(y_w)} {@code [n, 1]}
     * @param policyRejected {@code log pi(y_l)}
     * @param refChosen {@code log ref(y_w)} (no gradient)
     * @param refRejected {@code log ref(y_l)} (no gradient)
     * @param chosenLength token counts of the chosen responses
     * @param rejectedLength token counts of the rejected responses
     */
    public record Batch(Var policyChosen, Var policyRejected, Var refChosen, Var refRejected, float[] chosenLength,
            float[] rejectedLength) {
    }

    private LinkDpoLosses() {
    }

    /**
     * @param policy the policy
     * @param reference the frozen reference
     * @param prefs the preferences
     * @param normalize average token log-probabilities instead of summing them (SimPO)
     * @return the log-likelihoods
     */
    public static Batch batch(WindowLm policy, WindowLm reference, List<DpoTrainer.Preference> prefs,
            boolean normalize) {
        List<int[]> prompts = new ArrayList<>();
        List<int[]> completions = new ArrayList<>();
        int n = prefs.size();
        float[] lw = new float[n];
        float[] ll = new float[n];
        for (int i = 0; i < n; i++) {
            prompts.add(prefs.get(i).prompt());
            completions.add(prefs.get(i).chosen());
            lw[i] = prefs.get(i).chosen().length;
        }
        for (int i = 0; i < n; i++) {
            prompts.add(prefs.get(i).prompt());
            completions.add(prefs.get(i).rejected());
            ll[i] = prefs.get(i).rejected().length;
        }
        Tensor sum = WindowLm.sequenceSum(completions, normalize);
        Var seq = Var.of(sum).matmul(policy.logProbs(prompts, completions));
        Tensor ref = sum.matmul(reference.logProbs(prompts, completions).value());
        return new Batch(seq.slice(0, 0, n), seq.slice(0, n, 2 * n), Var.of(ref.slice(0, 0, n)),
                Var.of(ref.slice(0, n, 2 * n)), lw, ll);
    }

    /**
     * @param link a built-in link
     * @return its {@code psi} as a function of the log-ratio
     */
    public static UnaryOperator<Var> psi(Link link) {
        return switch (link) {
            case REVERSE_KL -> h -> h;
            case CHI_SQUARED -> h -> h.minimum(MAX_LOG_RATIO).exp().add(h);
            case SQUARED_LOG -> h -> h.mul(h.maximum(-MAX_LOG_RATIO).neg().exp());
            case JENSEN_SHANNON -> h -> {
                Var t = h.clamp(-MAX_LOG_RATIO, MAX_LOG_RATIO).exp();
                return t.mul(2).log().sub(t.add(1).log());
            };
        };
    }

    /**
     * The general f-DPO loss.
     *
     * @param policyChosen {@code log pi(y_w)}
     * @param policyRejected {@code log pi(y_l)}
     * @param refChosen {@code log ref(y_w)}
     * @param refRejected {@code log ref(y_l)}
     * @param beta the regularization strength
     * @param psi {@code f'(exp(h))} as a function of the log-ratio {@code h}
     * @return the mean loss
     */
    public static Var fDpo(Var policyChosen, Var policyRejected, Var refChosen, Var refRejected, float beta,
            UnaryOperator<Var> psi) {
        Var hw = policyChosen.sub(refChosen);
        Var hl = policyRejected.sub(refRejected);
        return PreferenceLosses.logSigmoid(psi.apply(hw).sub(psi.apply(hl)).mul(beta)).mean().neg();
    }

    /**
     * chiPO: f-DPO with the chi-squared-plus-KL link and the margin clipped to
     * {@code [-2 R, 2 R]}.
     *
     * @param policyChosen {@code log pi(y_w)}
     * @param policyRejected {@code log pi(y_l)}
     * @param refChosen {@code log ref(y_w)}
     * @param refRejected {@code log ref(y_l)}
     * @param beta the regularization strength
     * @param rMax the reward range {@code R} of the clip
     * @return the mean loss
     */
    public static Var chiSquared(Var policyChosen, Var policyRejected, Var refChosen, Var refRejected, float beta,
            float rMax) {
        UnaryOperator<Var> psi = psi(Link.CHI_SQUARED);
        Var m = psi.apply(policyChosen.sub(refChosen)).sub(psi.apply(policyRejected.sub(refRejected))).mul(beta)
                .clamp(-2 * rMax, 2 * rMax);
        return PreferenceLosses.logSigmoid(m).mean().neg();
    }

    /**
     * SquaredPO.
     *
     * @param policyChosen {@code log pi(y_w)}
     * @param policyRejected {@code log pi(y_l)}
     * @param refChosen {@code log ref(y_w)}
     * @param refRejected {@code log ref(y_l)}
     * @param beta the regularization strength
     * @return the mean loss
     */
    public static Var squared(Var policyChosen, Var policyRejected, Var refChosen, Var refRejected, float beta) {
        return fDpo(policyChosen, policyRejected, refChosen, refRejected, beta, psi(Link.SQUARED_LOG));
    }

    /**
     * @param b the log-likelihoods
     * @param link the link
     * @param beta the regularization strength
     * @return the f-DPO loss of the batch
     */
    public static Var loss(Batch b, Link link, float beta) {
        return fDpo(b.policyChosen(), b.policyRejected(), b.refChosen(), b.refRejected(), beta, psi(link));
    }
}
