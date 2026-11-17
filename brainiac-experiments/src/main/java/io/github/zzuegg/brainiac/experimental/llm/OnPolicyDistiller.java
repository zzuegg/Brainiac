package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Token-level knowledge distillation of a language model, on- or off-policy, with a choice
 * of divergence:
 * <ul>
 *   <li><b>Generalized knowledge distillation</b> (GKD, Agarwal et al. 2024,
 *       <a href="https://arxiv.org/abs/2306.13649">2306.13649</a>): with probability
 *       {@code lambda} a batch is <em>sampled by the student</em> and the teacher scores every
 *       position of those samples with its full next-token distribution; otherwise the batch
 *       is the supervised data. Training on the student's own prefixes removes the
 *       train/inference mismatch (exposure bias) of distilling only on gold prefixes. The
 *       divergence is the generalized Jensen-Shannon
 *       {@code JSD_beta = beta KL(p_T || M) + (1 - beta) KL(p_S || M)},
 *       {@code M = beta p_T + (1 - beta) p_S}, which interpolates between forward KL
 *       ({@code beta -> 0}, mean-seeking) and reverse KL ({@code beta -> 1}, mode-seeking).</li>
 *   <li><b>MiniLLM</b> (Gu et al. 2024, <a href="https://arxiv.org/abs/2306.08543">2306.08543</a>):
 *       the student minimizes the sequence-level <em>reverse</em> KL
 *       {@code KL(p_S || p_T)} on its own samples. Its gradient splits into the exact
 *       per-token reverse KL at each position (the single-step part) plus a policy gradient
 *       whose reward is the future log-ratio {@code sum_{s > t} log p_T(y_s) - log p_S(y_s)}
 *       ({@link Divergence#REVERSE_KL_SEQUENCE}). Mode-seeking: the student puts its mass on
 *       the teacher's main modes instead of smearing it over everything the teacher allows.</li>
 *   <li>The classic offline forward KL (Hinton et al. 2015), as in {@code DistillationLoss}, is
 *       {@link Divergence#FORWARD_KL} with {@code lambda = 0}.</li>
 * </ul>
 *
 * <pre>{@code
 * OnPolicyDistiller gkd = OnPolicyDistiller.of(student, teacher).divergence(Divergence.JSD, 0.5f)
 *         .onPolicyFraction(1).completions(maxTokens, eos);
 * gkd.step(prompts, gold, random);
 * }</pre>
 */
public final class OnPolicyDistiller {

    /** The token-level divergence between teacher and student. */
    public enum Divergence {
        /** {@code KL(p_T || p_S)}: mean-seeking, the classic distillation loss. */
        FORWARD_KL,
        /** {@code KL(p_S || p_T)} per token: mode-seeking. */
        REVERSE_KL,
        /** The generalized Jensen-Shannon divergence with weight {@code beta}. */
        JSD,
        /** MiniLLM: per-token reverse KL plus the policy gradient of the future log-ratio. */
        REVERSE_KL_SEQUENCE
    }

    private final WindowLm student;
    private final WindowLm teacher;
    private Divergence divergence = Divergence.JSD;
    private float beta = 0.5f;
    private double lambda = 1;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int maxNewTokens = 4;
    private int endToken = -1;
    private double temperature = 1;
    private float pgWeight = 1;
    private long teacherTokens;

    private OnPolicyDistiller(WindowLm student, WindowLm teacher) {
        this.student = Objects.requireNonNull(student);
        this.teacher = Objects.requireNonNull(teacher);
    }

    /**
     * @param student the model to train
     * @param teacher the frozen teacher, with the same vocabulary
     * @return the distiller
     */
    public static OnPolicyDistiller of(WindowLm student, WindowLm teacher) {
        return new OnPolicyDistiller(student, teacher);
    }

    /**
     * @param d the divergence
     * @param jsdBeta the JSD weight {@code beta} in {@code (0, 1)} (ignored by the others)
     * @return this distiller
     */
    public OnPolicyDistiller divergence(Divergence d, float jsdBeta) {
        this.divergence = Objects.requireNonNull(d);
        this.beta = jsdBeta;
        return this;
    }

    /**
     * @param fraction the share {@code lambda} of steps trained on student samples; the rest
     *     use the supervised completions
     * @return this distiller
     */
    public OnPolicyDistiller onPolicyFraction(double fraction) {
        this.lambda = fraction;
        return this;
    }

    /**
     * @param o the optimizer
     * @return this distiller
     */
    public OnPolicyDistiller optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param n the most tokens per student sample
     * @param end the end token
     * @return this distiller
     */
    public OnPolicyDistiller completions(int n, int end) {
        this.maxNewTokens = n;
        this.endToken = end;
        return this;
    }

    /**
     * @param t the student's sampling temperature
     * @return this distiller
     */
    public OnPolicyDistiller temperature(double t) {
        this.temperature = t;
        return this;
    }

    /**
     * @param w the weight of MiniLLM's policy-gradient (future log-ratio) term, 1 by default
     * @return this distiller
     */
    public OnPolicyDistiller policyGradientWeight(float w) {
        this.pgWeight = w;
        return this;
    }

    /** @return how many token positions the teacher has scored so far (its compute cost) */
    public long teacherTokens() {
        return teacherTokens;
    }

    /**
     * The mean per-position divergence between full next-token distributions.
     *
     * @param studentLogits the student's logits {@code [n, vocab]}
     * @param teacherLogits the teacher's logits {@code [n, vocab]}
     * @param d the divergence ({@code REVERSE_KL_SEQUENCE} gives the per-token reverse KL)
     * @param beta the JSD weight
     * @return the scalar divergence
     */
    public static Var divergence(Var studentLogits, Tensor teacherLogits, Divergence d, float beta) {
        int rows = studentLogits.shape().dim(0);
        Var logS = LlmTensors.logSoftmax(studentLogits);
        Tensor logT = LlmTensors.logSoftmax(Var.of(teacherLogits)).value();
        Tensor pT = logT.exp();
        Var pS = logS.exp();
        Var perRow = switch (d) {
            case FORWARD_KL -> logS.neg().add(logT).mul(pT).sum();
            case REVERSE_KL, REVERSE_KL_SEQUENCE -> logS.sub(logT).mul(pS).sum();
            case JSD -> {
                Var m = pS.mul(1 - beta).add(pT.mul(beta));
                Var logM = m.add(1e-12f).log();
                Var teacherPart = logM.neg().add(logT).mul(pT).sum().mul(beta);
                Var studentPart = logS.sub(logM).mul(pS).sum().mul(1 - beta);
                yield teacherPart.add(studentPart);
            }
        };
        return perRow.div(rows);
    }

    /**
     * One distillation step on a batch of prompts.
     *
     * @param prompts the prompts
     * @param gold the supervised completions for off-policy steps, or {@code null} to always
     *     sample from the student
     * @param random the random source
     * @return the divergence before the step
     */
    public float step(List<int[]> prompts, List<int[]> gold, SplittableRandom random) {
        boolean onPolicy = gold == null || random.nextDouble() < lambda;
        if (onPolicy && endToken < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken) first");
        }
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        int[][] samples = onPolicy
                ? student.sample(prompts.toArray(int[][]::new), maxNewTokens, endToken, temperature, random)
                : gold.toArray(int[][]::new);
        for (int i = 0; i < prompts.size(); i++) {
            if (samples[i].length > 0) {
                ps.add(prompts.get(i));
                cs.add(samples[i]);
            }
        }
        if (cs.isEmpty()) {
            return 0;
        }
        List<int[]> prefixes = prefixes(ps, cs);
        Var studentLogits = student.logits(prefixes);
        Tensor teacherLogits = teacher.logits(prefixes).value();
        teacherTokens += prefixes.size();
        Var loss = divergence(studentLogits, teacherLogits, divergence, beta);
        float value = loss.value().item();
        if (divergence == Divergence.REVERSE_KL_SEQUENCE && onPolicy) {
            loss = loss.add(futureRatioGradient(studentLogits, teacherLogits, cs).mul(pgWeight));
        }
        loss.backward();
        optimizer.step(student.parameters());
        return value;
    }

    /**
     * MiniLLM's long-term part: REINFORCE on {@code log p_S(y_t)} with the detached reward
     * {@code R_{t+1} = sum_{s > t} (log p_T(y_s) - log p_S(y_s))}, whitened over the batch.
     */
    private static Var futureRatioGradient(Var studentLogits, Tensor teacherLogits, List<int[]> cs) {
        int[] targets = new int[studentLogits.shape().dim(0)];
        int k = 0;
        for (int[] c : cs) {
            for (int t : c) {
                targets[k++] = t;
            }
        }
        Var logpS = LlmTensors.pick(LlmTensors.logSoftmax(studentLogits), targets);
        float[] s = LlmTensors.values(logpS);
        float[] t = LlmTensors.values(LlmTensors.pick(LlmTensors.logSoftmax(Var.of(teacherLogits)), targets));
        float[] reward = new float[targets.length];
        k = 0;
        for (int[] c : cs) {
            double future = 0;
            for (int i = c.length - 1; i >= 0; i--) {
                reward[k + i] = (float) future;
                future += t[k + i] - s[k + i];
            }
            k += c.length;
        }
        double mean = LlmTensors.mean(reward);
        double var = 0;
        for (float r : reward) {
            var += (r - mean) * (r - mean);
        }
        double std = Math.sqrt(var / Math.max(1, reward.length - 1)) + 1e-6;
        for (int i = 0; i < reward.length; i++) {
            reward[i] = (float) ((reward[i] - mean) / std);
        }
        return logpS.mul(LlmTensors.column(reward)).mean().neg();
    }

    /**
     * @param prompts the prompts
     * @param completions what followed each prompt
     * @return the prefix before every completion token, in {@code logProbs} order
     */
    public static List<int[]> prefixes(List<int[]> prompts, List<int[]> completions) {
        List<int[]> prefixes = new ArrayList<>();
        for (int s = 0; s < prompts.size(); s++) {
            int[] p = prompts.get(s);
            int[] c = completions.get(s);
            for (int t = 0; t < c.length; t++) {
                int[] prefix = Arrays.copyOf(p, p.length + t);
                System.arraycopy(c, 0, prefix, p.length, t);
                prefixes.add(prefix);
            }
        }
        return prefixes;
    }
}
