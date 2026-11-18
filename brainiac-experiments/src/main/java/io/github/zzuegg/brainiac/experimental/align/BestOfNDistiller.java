package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * BOND, best-of-N distillation (Sessa et al. 2024, <a href="https://arxiv.org/abs/2407.14622">
 * arXiv:2407.14622</a>): train the policy to <em>be</em> the best-of-N sampler of a reference
 * policy, so that one sample costs what one sample costs but is as good as the best of N. The
 * best-of-N distribution has a closed form in the reference's reward quantiles,
 * {@code p_BoN(y) ~ p_ref(y) F(r(y))^(N - 1)} with {@code F(r) = P_{y' ~ ref}(r(y') <= r)}; the
 * quantile is estimated from {@link #referenceSamples} reference samples per prompt, so only
 * the ranking of rewards matters, not their scale. The policy is fitted by distribution
 * matching with the Jeffreys divergence {@code (1 - beta) KL(p_BoN || pi) + beta KL(pi || p_BoN)}:
 * <ul>
 *   <li>the forward (mode-covering) term is a weighted likelihood on the reference samples,
 *       each weighted by its self-normalized best-of-N importance weight {@code F(r)^(N-1)};</li>
 *   <li>the backward (mode-seeking) term is a policy gradient on the policy's own samples with
 *       the reward {@code log p_ref(y) + (N - 1) log F(r(y)) - log pi(y)} and a leave-one-out
 *       baseline.</li>
 * </ul>
 * {@link #iterated} switches to J-BOND's iterated variant: best-of-2 against a moving anchor
 * (an exponential moving average of the policy's weights), so that after {@code t} anchor
 * updates the effective N grows without ever needing N samples per prompt.
 */
public final class BestOfNDistiller {

    /**
     * What one step did.
     *
     * @param forwardLoss the forward-KL (weighted likelihood) term
     * @param backwardLoss the backward-KL (policy gradient) surrogate
     * @param meanReward the mean reward of the policy's samples
     */
    public record Stats(double forwardLoss, double backwardLoss, double meanReward) {
    }

    private final WindowLm policy;
    private final WindowLm reference;
    private WindowLm anchor;
    private float anchorRate;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int n = 16;
    private float beta = 0.5f;
    private int refSamples = 32;
    private int policySamples = 8;
    private int maxTokens = 4;
    private int end = -1;
    private long samplesDrawn;

    private BestOfNDistiller(WindowLm policy, WindowLm reference) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = Objects.requireNonNull(reference);
    }

    /**
     * @param policy the model to train
     * @param reference the reference whose best-of-N distribution is distilled (not trained)
     * @return the distiller
     */
    public static BestOfNDistiller of(WindowLm policy, WindowLm reference) {
        return new BestOfNDistiller(policy, reference);
    }

    /**
     * @param bestOf the N of best-of-N
     * @return this distiller
     */
    public BestOfNDistiller bestOf(int bestOf) {
        this.n = bestOf;
        return this;
    }

    /**
     * @param b the weight of the backward KL in the Jeffreys divergence (0: forward only, 1:
     *     backward only)
     * @return this distiller
     */
    public BestOfNDistiller jeffreys(float b) {
        this.beta = b;
        return this;
    }

    /**
     * @param m reference samples per prompt for the quantile estimate (also the forward term's
     *     samples)
     * @return this distiller
     */
    public BestOfNDistiller referenceSamples(int m) {
        this.refSamples = m;
        return this;
    }

    /**
     * @param k policy samples per prompt for the backward term
     * @return this distiller
     */
    public BestOfNDistiller policySamples(int k) {
        this.policySamples = k;
        return this;
    }

    /**
     * @param max the most tokens per response
     * @param endToken the end token
     * @return this distiller
     */
    public BestOfNDistiller completions(int max, int endToken) {
        this.maxTokens = max;
        this.end = endToken;
        return this;
    }

    /**
     * @param o the optimizer
     * @return this distiller
     */
    public BestOfNDistiller optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * J-BOND: distil best-of-2 of a moving anchor instead of best-of-N of the fixed reference.
     * The anchor starts as a copy of the reference and moves towards the policy by {@code rate}
     * after every step.
     *
     * @param rate the anchor's EMA rate, for example 0.05
     * @return this distiller
     */
    public BestOfNDistiller iterated(float rate) {
        this.anchor = reference.copy();
        this.anchorRate = rate;
        this.n = 2;
        return this;
    }

    /** @return responses sampled so far (reference and policy), the training cost */
    public long samplesDrawn() {
        return samplesDrawn;
    }

    /**
     * The empirical reward quantile of each value among reference rewards, counting ties as
     * half, clamped away from 0.
     *
     * @param values the rewards to place
     * @param referenceRewards the reference rewards
     * @return {@code F(r)} for each value
     */
    public static double[] quantiles(float[] values, float[] referenceRewards) {
        double[] q = new double[values.length];
        int m = referenceRewards.length;
        for (int i = 0; i < values.length; i++) {
            double below = 0;
            for (float r : referenceRewards) {
                below += r < values[i] ? 1 : r == values[i] ? 0.5 : 0;
            }
            q[i] = Math.max(0.5 / (m + 1), (below + 0.5) / (m + 1));
        }
        return q;
    }

    /**
     * One distillation step.
     *
     * @param prompts draws a prompt
     * @param promptsPerStep prompts per step
     * @param reward the reward (only its ranking matters)
     * @param random the random source
     * @return what happened
     */
    public Stats step(Function<SplittableRandom, int[]> prompts, int promptsPerStep, GrpoTrainer.Reward reward,
            SplittableRandom random) {
        if (end < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken) first");
        }
        WindowLm base = anchor == null ? reference : anchor;
        List<int[]> fp = new ArrayList<>();
        List<int[]> fc = new ArrayList<>();
        List<Float> fw = new ArrayList<>();
        List<int[]> bp = new ArrayList<>();
        List<int[]> bc = new ArrayList<>();
        List<Float> ba = new ArrayList<>();
        double rewardSum = 0;
        int rewardCount = 0;
        for (int p = 0; p < promptsPerStep; p++) {
            int[] prompt = prompts.apply(random);
            int[][] rp = new int[refSamples][];
            Arrays.fill(rp, prompt);
            int[][] refOut = base.sample(rp, maxTokens, end, 1.0, random);
            float[] refR = new float[refSamples];
            for (int i = 0; i < refSamples; i++) {
                refR[i] = reward.score(prompt, refOut[i]);
            }
            // forward KL: self-normalized best-of-N weights on the reference samples
            double[] fq = quantiles(refR, refR);
            double[] w = new double[refSamples];
            double wsum = 0;
            for (int i = 0; i < refSamples; i++) {
                w[i] = Math.pow(fq[i], n - 1);
                wsum += w[i];
            }
            for (int i = 0; i < refSamples; i++) {
                if (refOut[i].length > 0) {
                    fp.add(prompt);
                    fc.add(refOut[i]);
                    fw.add((float) (w[i] / wsum));
                }
            }
            // backward KL: policy samples, reward log p_ref + (N - 1) log F - log pi
            int[][] pp = new int[policySamples][];
            Arrays.fill(pp, prompt);
            int[][] polOut = policy.sample(pp, maxTokens, end, 1.0, random);
            float[] polR = new float[policySamples];
            for (int i = 0; i < policySamples; i++) {
                polR[i] = reward.score(prompt, polOut[i]);
                rewardSum += polR[i];
                rewardCount++;
            }
            double[] pq = quantiles(polR, refR);
            List<int[]> live = new ArrayList<>();
            List<Integer> idx = new ArrayList<>();
            for (int i = 0; i < policySamples; i++) {
                if (polOut[i].length > 0) {
                    live.add(polOut[i]);
                    idx.add(i);
                }
            }
            if (live.size() < 2) {
                continue;
            }
            List<int[]> lp = new ArrayList<>();
            for (int i = 0; i < live.size(); i++) {
                lp.add(prompt);
            }
            float[] logRef = base.sequenceLogProbs(lp, live).value().toArray();
            float[] logPi = policy.sequenceLogProbs(lp, live).value().toArray();
            double[] target = new double[live.size()];
            double tsum = 0;
            for (int i = 0; i < live.size(); i++) {
                target[i] = logRef[i] + (n - 1) * Math.log(pq[idx.get(i)]) - logPi[i];
                tsum += target[i];
            }
            for (int i = 0; i < live.size(); i++) {
                double baseline = (tsum - target[i]) / (live.size() - 1);
                bp.add(prompt);
                bc.add(live.get(i));
                ba.add((float) (target[i] - baseline));
            }
            samplesDrawn += refSamples + policySamples;
        }
        double fwd = 0;
        double bwd = 0;
        Var loss = null;
        if (!fc.isEmpty() && beta < 1) {
            Var logp = policy.sequenceLogProbs(fp, fc);
            Var f = logp.mul(column(fw)).sum().neg().mul(1f / promptsPerStep);
            fwd = f.value().item();
            loss = f.mul(1 - beta);
        }
        if (!bc.isEmpty() && beta > 0) {
            Var logp = policy.sequenceLogProbs(bp, bc);
            Var b = logp.mul(column(ba)).mean().neg();
            bwd = b.value().item();
            loss = loss == null ? b.mul(beta) : loss.add(b.mul(beta));
        }
        if (loss != null) {
            loss.backward();
            optimizer.step(policy.parameters());
        }
        if (anchor != null) {
            WeightAveragedReward.interpolate(anchor.parameters(), policy.parameters(), anchorRate,
                    anchor.parameters());
        }
        return new Stats(fwd, bwd, rewardCount == 0 ? 0 : rewardSum / rewardCount);
    }

    private static Tensor column(List<Float> values) {
        float[] v = new float[values.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = values.get(i);
        }
        return Tensor.of(Shape.of(v.length, 1), v);
    }
}
