package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;
import java.util.Objects;

/**
 * Length-regularized DPO (R-DPO, Park et al., ACL Findings 2024,
 * <a href="https://arxiv.org/abs/2403.19159">arXiv:2403.19159</a>). Preference data often favour
 * the longer answer independently of quality, and DPO exploits this: the implicit reward
 * rises with length, so the tuned policy writes ever longer answers without getting better.
 * R-DPO derives DPO from an RLHF objective with a length penalty, which adds a length-difference
 * term to the margin, {@code -log sigmoid(beta h + alpha (|y_w| - |y_l|))}: a pair won by the
 * longer answer counts as already partly explained by its length, so its gradient shrinks, and
 * length alone no longer pays.
 */
public final class LengthRegularizedDpo {

    private final WindowLm policy;
    private WindowLm reference;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float beta = 0.1f;
    private float alpha = 0.05f;

    private LengthRegularizedDpo(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = policy.copy();
    }

    /**
     * @param policy the model to train; a frozen copy becomes the reference
     * @return the trainer
     */
    public static LengthRegularizedDpo of(WindowLm policy) {
        return new LengthRegularizedDpo(policy);
    }

    /**
     * @param b the DPO inverse temperature
     * @param a the length penalty per token of length difference
     * @return this trainer
     */
    public LengthRegularizedDpo strength(float b, float a) {
        this.beta = b;
        this.alpha = a;
        return this;
    }

    /**
     * @param o the optimizer
     * @return this trainer
     */
    public LengthRegularizedDpo optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param ref the frozen reference
     * @return this trainer
     */
    public LengthRegularizedDpo reference(WindowLm ref) {
        this.reference = Objects.requireNonNull(ref);
        return this;
    }

    /**
     * The R-DPO loss.
     *
     * @param policyChosen {@code log pi(y_w)} {@code [n, 1]}
     * @param policyRejected {@code log pi(y_l)}
     * @param refChosen {@code log ref(y_w)}
     * @param refRejected {@code log ref(y_l)}
     * @param chosenLength token counts of the chosen responses
     * @param rejectedLength token counts of the rejected responses
     * @param beta the inverse temperature
     * @param alpha the length penalty
     * @return the mean loss
     */
    public static Var loss(Var policyChosen, Var policyRejected, Var refChosen, Var refRejected, float[] chosenLength,
            float[] rejectedLength, float beta, float alpha) {
        int n = chosenLength.length;
        float[] d = new float[n];
        for (int i = 0; i < n; i++) {
            d[i] = alpha * (chosenLength[i] - rejectedLength[i]);
        }
        Var h = PreferenceLosses.margin(policyChosen, policyRejected, refChosen, refRejected);
        return PreferenceLosses.logSigmoid(h.mul(beta).add(Tensor.of(h.shape(), reshape(d, h.shape())))).mean()
                .neg();
    }

    private static float[] reshape(float[] d, Shape shape) {
        if (shape.size() != d.length) {
            throw new IllegalArgumentException("one length difference per pair");
        }
        return d;
    }

    /**
     * One optimizer step.
     *
     * @param prefs the preferences
     * @return the loss before the step
     */
    public float step(List<DpoTrainer.Preference> prefs) {
        LinkDpoLosses.Batch b = LinkDpoLosses.batch(policy, reference, prefs, false);
        Var l = loss(b.policyChosen(), b.policyRejected(), b.refChosen(), b.refRejected(), b.chosenLength(),
                b.rejectedLength(), beta, alpha);
        float v = l.value().item();
        l.backward();
        optimizer.step(policy.parameters());
        return v;
    }
}
