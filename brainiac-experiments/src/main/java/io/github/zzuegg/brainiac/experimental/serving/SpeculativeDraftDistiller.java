package io.github.zzuegg.brainiac.experimental.serving;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.OnPolicyDistiller;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * DistillSpec: distilling the draft model of speculative decoding for acceptance (Zhou et al.,
 * ICLR 2024, <a href="https://arxiv.org/abs/2310.08461">arXiv:2310.08461</a>). The speed-up of
 * speculative sampling is set by how often the target accepts the draft's tokens, and the
 * acceptance probability of one token is {@code sum_x min(p(x), q(x)) = 1 - TVD(p, q)}. So
 * DistillSpec trains the draft {@code q} to minimise the total variation distance to the target
 * {@code p} (or another divergence: forward or reverse KL, JSD, or a skew of the two KLs) on the
 * draft's <em>own</em> samples, the prefixes it will meet while drafting, rather than on
 * ground-truth text.
 *
 * <p>The sampling loop and the KL family reuse {@link OnPolicyDistiller}; this class adds the
 * total variation distance and the skewed KL, the on-policy share, and the acceptance
 * measurements: {@link #acceptance} gives the expected per-token acceptance rate of a draft
 * along its own samples, at a temperature or greedily.
 */
public final class SpeculativeDraftDistiller {

    /** The divergence minimised. */
    public enum Divergence {
        /** {@code 0.5 sum |p - q|}: exactly one minus the acceptance rate. */
        TVD,
        /** {@code KL(p || q)}. */
        FORWARD_KL,
        /** {@code KL(q || p)}. */
        REVERSE_KL,
        /** {@code (1 - skew) KL(p || q) + skew KL(q || p)}. */
        SKEW_KL,
        /** The generalized Jensen-Shannon divergence with weight {@code 1/2}. */
        JSD
    }

    private final WindowLm draft;
    private final WindowLm target;
    private Divergence divergence = Divergence.TVD;
    private float skew = 0.5f;
    private double onPolicy = 1;
    private double temperature = 1;
    private int maxNew = 6;
    private int end = -1;
    private Optimizer optimizer = Optimizer.adam(3e-3f);

    private SpeculativeDraftDistiller(WindowLm draft, WindowLm target) {
        this.draft = Objects.requireNonNull(draft, "draft");
        this.target = Objects.requireNonNull(target, "target");
    }

    /**
     * @param draft the draft model, trained in place
     * @param target the frozen target model
     * @return the distiller
     */
    public static SpeculativeDraftDistiller of(WindowLm draft, WindowLm target) {
        return new SpeculativeDraftDistiller(draft, target);
    }

    /**
     * @param d the divergence
     * @param skewWeight the reverse-KL weight of {@code SKEW_KL}
     * @return this distiller
     */
    public SpeculativeDraftDistiller divergence(Divergence d, float skewWeight) {
        this.divergence = Objects.requireNonNull(d);
        this.skew = skewWeight;
        return this;
    }

    /**
     * @param share the probability that a batch is sampled by the draft (otherwise the given
     *     data is used)
     * @return this distiller
     */
    public SpeculativeDraftDistiller onPolicyShare(double share) {
        this.onPolicy = share;
        return this;
    }

    /**
     * @param n the tokens sampled per prompt
     * @param endToken the end token, or -1
     * @param t the sampling temperature of the draft
     * @return this distiller
     */
    public SpeculativeDraftDistiller sampling(int n, int endToken, double t) {
        this.maxNew = n;
        this.end = endToken;
        this.temperature = t;
        return this;
    }

    /**
     * @param o the draft's optimizer
     * @return this distiller
     */
    public SpeculativeDraftDistiller optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * The divergence between the target's and the draft's distributions, averaged over rows.
     *
     * @param draftLogits the draft's logits {@code [n, vocab]}
     * @param targetLogits the target's logits {@code [n, vocab]}
     * @param d the divergence
     * @param skew the reverse-KL weight of {@code SKEW_KL}
     * @return the scalar loss
     */
    public static Var loss(Var draftLogits, Tensor targetLogits, Divergence d, float skew) {
        return switch (d) {
            case FORWARD_KL -> OnPolicyDistiller.divergence(draftLogits, targetLogits,
                    OnPolicyDistiller.Divergence.FORWARD_KL, 0);
            case REVERSE_KL -> OnPolicyDistiller.divergence(draftLogits, targetLogits,
                    OnPolicyDistiller.Divergence.REVERSE_KL, 0);
            case JSD -> OnPolicyDistiller.divergence(draftLogits, targetLogits, OnPolicyDistiller.Divergence.JSD,
                    0.5f);
            case SKEW_KL -> OnPolicyDistiller.divergence(draftLogits, targetLogits,
                    OnPolicyDistiller.Divergence.FORWARD_KL, 0).mul(1 - skew).add(OnPolicyDistiller.divergence(
                    draftLogits, targetLogits, OnPolicyDistiller.Divergence.REVERSE_KL, 0).mul(skew));
            case TVD -> {
                Var q = Activation.LOG_SOFTMAX.forward(draftLogits).exp();
                Tensor p = Activation.LOG_SOFTMAX.forward(Var.of(targetLogits)).value().exp();
                yield q.sub(p).abs().sum().mul(0.5f / draftLogits.shape().dim(0));
            }
        };
    }

    /**
     * One step: the draft samples continuations of the prompts (or, off-policy, the data is
     * used), the target scores every prefix, and the draft moves toward the target.
     *
     * @param prompts the prompts
     * @param data supervised continuations for off-policy batches, or {@code null}
     * @param random the random source
     * @return the loss before the step
     */
    public float step(List<int[]> prompts, List<int[]> data, SplittableRandom random) {
        boolean sample = data == null || random.nextDouble() < onPolicy;
        int[][] outs = sample ? draft.sample(prompts.toArray(int[][]::new), maxNew, end, temperature, random)
                : data.toArray(int[][]::new);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            if (outs[i].length > 0) {
                ps.add(prompts.get(i));
                cs.add(outs[i]);
            }
        }
        if (cs.isEmpty()) {
            return 0;
        }
        List<int[]> prefixes = OnPolicyDistiller.prefixes(ps, cs);
        Var draftLogits = draft.logits(prefixes);
        Tensor targetLogits = Var.noGrad(() -> target.logits(prefixes)).value();
        Var l = loss(draftLogits, targetLogits, divergence, skew);
        float value = l.value().item();
        l.backward();
        optimizer.step(draft.parameters());
        return value;
    }

    /**
     * The expected acceptance rate of speculative sampling with a draft, along the draft's own
     * continuations: at temperature {@code t > 0}, the mean of {@code sum_x min(p_t(x), q_t(x))}
     * over the positions of draft samples; at {@code t = 0} (greedy), the share of positions
     * where both argmaxes agree along the draft's greedy continuation.
     *
     * @param draft the draft
     * @param target the target
     * @param prompts the prompts
     * @param tokens tokens drafted per prompt
     * @param t the temperature, 0 for greedy
     * @param random the random source
     * @return the mean acceptance probability per drafted token
     */
    public static double acceptance(WindowLm draft, WindowLm target, List<int[]> prompts, int tokens, double t,
            SplittableRandom random) {
        double sum = 0;
        int n = 0;
        for (int[] prompt : prompts) {
            int[] seq = prompt.clone();
            for (int i = 0; i < tokens; i++) {
                double[] q = probabilities(draft.logits(seq), t);
                double[] p = probabilities(target.logits(seq), t);
                double a = 0;
                for (int x = 0; x < p.length; x++) {
                    a += Math.min(p[x], q[x]);
                }
                sum += a;
                n++;
                int next = sample(q, random);
                seq = java.util.Arrays.copyOf(seq, seq.length + 1);
                seq[seq.length - 1] = next;
            }
        }
        return sum / n;
    }

    /**
     * A temperature softmax; {@code t = 0} gives a one-hot at the argmax.
     *
     * @param logits the logits
     * @param t the temperature
     * @return the probabilities
     */
    public static double[] probabilities(float[] logits, double t) {
        double[] p = new double[logits.length];
        if (t <= 0) {
            int b = 0;
            for (int i = 1; i < logits.length; i++) {
                if (logits[i] > logits[b]) {
                    b = i;
                }
            }
            p[b] = 1;
            return p;
        }
        double max = Double.NEGATIVE_INFINITY;
        for (float l : logits) {
            max = Math.max(max, l / t);
        }
        double z = 0;
        for (int i = 0; i < p.length; i++) {
            p[i] = Math.exp(logits[i] / t - max);
            z += p[i];
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= z;
        }
        return p;
    }

    static int sample(double[] p, SplittableRandom random) {
        double u = random.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }

    @Override
    public String toString() {
        return "DistillSpec(" + divergence + ", on-policy " + onPolicy + ")";
    }
}
