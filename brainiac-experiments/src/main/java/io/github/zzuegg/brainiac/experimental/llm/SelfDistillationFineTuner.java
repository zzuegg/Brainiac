package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Self-distillation fine-tuning (SDFT): learn a new skill from demonstrations without the
 * forgetting of supervised fine-tuning. The model <em>conditioned on a demonstration in its
 * context</em> already does roughly the right thing (in-context learning); that conditioned
 * model is the teacher, and the plain, unconditioned model is the student. Each step:
 *
 * <ol>
 *   <li>the student samples completions for the prompts (on-policy);</li>
 *   <li>the teacher, the same weights with the demonstration prepended, scores every
 *       position of those samples with its full next-token distribution (no gradient);</li>
 *   <li>the student minimizes the per-token divergence to the teacher (reverse KL by
 *       default, via {@link OnPolicyDistiller#divergence}).</li>
 * </ol>
 *
 * <p>Unlike SFT on the demonstrations' fixed tokens, the targets are the model's own
 * distribution shifted only as far as the demonstration shifts it, and they are evaluated on
 * the student's own samples, so the update stays close to the base (the mechanism of "RL's
 * Razor"), yet no reward function is needed.
 *
 * <p>Paper: "Self-Distillation Enables Continual Learning" (2026),
 * <a href="https://arxiv.org/abs/2601.19897">arXiv:2601.19897</a>.
 */
public final class SelfDistillationFineTuner {

    private final WindowLm model;
    private Optimizer optimizer = Optimizer.adam(1e-3f);
    private OnPolicyDistiller.Divergence divergence = OnPolicyDistiller.Divergence.REVERSE_KL;
    private int maxNewTokens = 4;
    private int endToken = -1;
    private double temperature = 1;
    private WindowLm frozenTeacher;

    private SelfDistillationFineTuner(WindowLm model) {
        this.model = Objects.requireNonNull(model);
    }

    /**
     * @param model the model to fine-tune; it is also its own teacher
     * @return the fine-tuner
     */
    public static SelfDistillationFineTuner of(WindowLm model) {
        return new SelfDistillationFineTuner(model);
    }

    /**
     * @param o the optimizer
     * @return this
     */
    public SelfDistillationFineTuner optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param d the token-level divergence (reverse KL by default)
     * @return this
     */
    public SelfDistillationFineTuner divergence(OnPolicyDistiller.Divergence d) {
        this.divergence = Objects.requireNonNull(d);
        return this;
    }

    /**
     * @param n the most tokens per student sample
     * @param end the end token
     * @return this
     */
    public SelfDistillationFineTuner completions(int n, int end) {
        this.maxNewTokens = n;
        this.endToken = end;
        return this;
    }

    /**
     * @param t the student's sampling temperature
     * @return this
     */
    public SelfDistillationFineTuner temperature(double t) {
        this.temperature = t;
        return this;
    }

    /**
     * Uses a frozen copy of the starting model as the conditioned teacher instead of the
     * current weights (the default, which lets the teacher improve with the student).
     *
     * @return this
     */
    public SelfDistillationFineTuner frozenTeacher() {
        this.frozenTeacher = model.copy();
        return this;
    }

    /**
     * One self-distillation step.
     *
     * @param prompts the prompts the student answers
     * @param demonstrations for each prompt, the demonstration tokens the teacher sees before it
     * @param random the sampling source
     * @return the divergence before the step
     */
    public float step(List<int[]> prompts, List<int[]> demonstrations, SplittableRandom random) {
        if (endToken < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken) first");
        }
        int[][] samples = model.sample(prompts.toArray(int[][]::new), maxNewTokens, endToken, temperature, random);
        List<int[]> ps = new ArrayList<>();
        List<int[]> teacherPs = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            if (samples[i].length > 0) {
                ps.add(prompts.get(i));
                teacherPs.add(concat(demonstrations.get(i), prompts.get(i)));
                cs.add(samples[i]);
            }
        }
        if (cs.isEmpty()) {
            return 0;
        }
        WindowLm teacher = frozenTeacher != null ? frozenTeacher : model;
        Tensor teacherLogits = teacher.logits(OnPolicyDistiller.prefixes(teacherPs, cs)).value();
        Var loss = OnPolicyDistiller.divergence(model.logits(OnPolicyDistiller.prefixes(ps, cs)), teacherLogits,
                divergence, 0.5f);
        float value = loss.value().item();
        for (Var p : model.parameters()) {
            p.zeroGrad();
        }
        loss.backward();
        optimizer.step(model.parameters());
        return value;
    }

    /**
     * Concatenates token arrays.
     *
     * @param a the first part
     * @param b the second part
     * @return {@code a} then {@code b}
     */
    public static int[] concat(int[] a, int[] b) {
        int[] out = new int[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
