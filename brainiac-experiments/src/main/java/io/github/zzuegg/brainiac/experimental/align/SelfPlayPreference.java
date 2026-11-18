package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
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
 * Preference optimization against a general preference model, without a reward: the policy
 * plays a two-player game in which each player proposes a response and the preference model
 * decides who wins, and seeks its Nash equilibrium, a policy that no other policy is preferred
 * to. Unlike Bradley-Terry RLHF and DPO this does not assume preferences are transitive, so a
 * cyclic majority (annotator types that prefer rock over scissors, scissors over paper and
 * paper over rock) yields a mixed policy instead of a collapse onto one style.
 * <ul>
 *   <li>{@link #sppoIteration}: Self-Play Preference Optimization (Wu et al. 2024,
 *       <a href="https://arxiv.org/abs/2405.00675">arXiv:2405.00675</a>). Sample {@code K}
 *       responses per prompt from the current policy {@code pi_t}, estimate each one's win rate
 *       against {@code pi_t} with the preference model, {@code P(y > pi_t) = mean_j P(y > y_j)},
 *       and regress the log-ratio onto it:
 *       {@code (log pi(y) - log pi_t(y) - eta (P(y > pi_t) - 1/2))^2}. Each iteration is one
 *       step of multiplicative weights, whose average converges to the Nash equilibrium.</li>
 *   <li>{@link #nashMdStep}: Nash-MD (Munos et al., ICML 2024,
 *       <a href="https://arxiv.org/abs/2312.00886">arXiv:2312.00886</a>), mirror descent towards
 *       a regularized opponent, in its policy-gradient form: the opponent samples from the
 *       geometric mixture {@code pi^(1 - mu) ref^mu} (mixed per token in logit space), the policy
 *       is rewarded by {@code P(y > y') - 1/2} and kept near the reference by a KL penalty.</li>
 * </ul>
 */
public final class SelfPlayPreference {

    /** A general preference model over responses to a prompt. */
    @FunctionalInterface
    public interface Preference {
        /**
         * @param prompt the prompt
         * @param candidates responses to it
         * @return {@code P[i][j] = P(candidate i > candidate j)}
         */
        double[][] matrix(int[] prompt, List<int[]> candidates);
    }

    private final WindowLm policy;
    private final WindowLm reference;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private int samples = 8;
    private int maxTokens = 4;
    private int end = -1;
    private float eta = 2;
    private float mixture = 0.1f;
    private float kl = 0.02f;

    private SelfPlayPreference(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = policy.copy();
    }

    /**
     * @param policy the model to train; a frozen copy becomes the reference
     * @return the trainer
     */
    public static SelfPlayPreference of(WindowLm policy) {
        return new SelfPlayPreference(policy);
    }

    /**
     * @param o the optimizer
     * @return this trainer
     */
    public SelfPlayPreference optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param k responses sampled per prompt
     * @return this trainer
     */
    public SelfPlayPreference samples(int k) {
        this.samples = k;
        return this;
    }

    /**
     * @param max the most tokens per response
     * @param endToken the end token
     * @return this trainer
     */
    public SelfPlayPreference completions(int max, int endToken) {
        this.maxTokens = max;
        this.end = endToken;
        return this;
    }

    /**
     * @param e SPPO's step size {@code eta} (the paper's {@code 1 / beta})
     * @return this trainer
     */
    public SelfPlayPreference eta(float e) {
        this.eta = e;
        return this;
    }

    /**
     * @param mu the reference's weight in Nash-MD's geometric-mixture opponent
     * @param beta the KL penalty towards the reference
     * @return this trainer
     */
    public SelfPlayPreference nashMd(float mu, float beta) {
        this.mixture = mu;
        this.kl = beta;
        return this;
    }

    /**
     * Win rates of each candidate against the whole set (the empirical opponent), the
     * diagonal excluded.
     *
     * @param p the preference matrix
     * @return {@code mean_{j != i} P[i][j]}
     */
    public static double[] winRates(double[][] p) {
        int n = p.length;
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += i == j ? 0 : p[i][j];
            }
            w[i] = n > 1 ? s / (n - 1) : 0.5;
        }
        return w;
    }

    /**
     * One SPPO iteration: sample from the current policy, score win rates, then fit the
     * squared loss for {@code epochs} steps against the frozen snapshot {@code pi_t}.
     *
     * @param prompts draws a prompt
     * @param promptsPerIteration prompts per iteration
     * @param preference the preference model
     * @param epochs gradient steps on the iteration's data
     * @param random the random source
     * @return the mean estimated win rate of the samples (about 1/2 by construction)
     */
    public double sppoIteration(Function<SplittableRandom, int[]> prompts, int promptsPerIteration,
            Preference preference, int epochs, SplittableRandom random) {
        check();
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<Float> target = new ArrayList<>();
        double mean = 0;
        int count = 0;
        for (int q = 0; q < promptsPerIteration; q++) {
            int[] prompt = prompts.apply(random);
            int[][] rep = new int[samples][];
            Arrays.fill(rep, prompt);
            List<int[]> out = new ArrayList<>(Arrays.asList(policy.sample(rep, maxTokens, end, 1.0, random)));
            out.removeIf(c -> c.length == 0);
            if (out.size() < 2) {
                continue;
            }
            double[] w = winRates(preference.matrix(prompt, out));
            for (int i = 0; i < out.size(); i++) {
                ps.add(prompt);
                cs.add(out.get(i));
                target.add((float) (eta * (w[i] - 0.5)));
                mean += w[i];
                count++;
            }
        }
        if (cs.isEmpty()) {
            return 0.5;
        }
        Tensor old = policy.sequenceLogProbs(ps, cs).value();
        float[] t = new float[target.size()];
        for (int i = 0; i < t.length; i++) {
            t[i] = target.get(i);
        }
        Tensor tt = Tensor.of(Shape.of(t.length, 1), t);
        for (int e = 0; e < epochs; e++) {
            Var loss = policy.sequenceLogProbs(ps, cs).sub(old).sub(tt).square().mean();
            loss.backward();
            optimizer.step(policy.parameters());
        }
        return mean / count;
    }

    /**
     * Samples from the per-token geometric mixture {@code pi^(1 - mu) ref^mu}.
     *
     * @param prompts the prompts
     * @param random the random source
     * @return the responses
     */
    public int[][] sampleMixture(int[][] prompts, SplittableRandom random) {
        int n = prompts.length;
        int[][] seqs = new int[n][];
        int[] len = new int[n];
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            seqs[i] = Arrays.copyOf(prompts[i], prompts[i].length + maxTokens);
            len[i] = prompts[i].length;
        }
        int vocab = policy.vocabSize();
        for (int t = 0; t < maxTokens; t++) {
            List<int[]> live = new ArrayList<>();
            List<Integer> who = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i]) {
                    live.add(Arrays.copyOf(seqs[i], len[i]));
                    who.add(i);
                }
            }
            if (live.isEmpty()) {
                break;
            }
            float[] a = Var.noGrad(() -> policy.logits(live)).value().toArray();
            float[] b = Var.noGrad(() -> reference.logits(live)).value().toArray();
            for (int j = 0; j < who.size(); j++) {
                double[] p = new double[vocab];
                double max = Double.NEGATIVE_INFINITY;
                for (int v = 0; v < vocab; v++) {
                    p[v] = (1 - mixture) * a[j * vocab + v] + mixture * b[j * vocab + v];
                    max = Math.max(max, p[v]);
                }
                double sum = 0;
                for (int v = 0; v < vocab; v++) {
                    p[v] = Math.exp(p[v] - max);
                    sum += p[v];
                }
                double u = random.nextDouble() * sum;
                int tok = vocab - 1;
                for (int v = 0; v < vocab; v++) {
                    u -= p[v];
                    if (u <= 0) {
                        tok = v;
                        break;
                    }
                }
                int i = who.get(j);
                seqs[i][len[i]++] = tok;
                done[i] |= tok == end;
            }
        }
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = Arrays.copyOfRange(seqs[i], prompts[i].length, len[i]);
        }
        return out;
    }

    /**
     * One Nash-MD policy-gradient step: the policy's samples play the mixture opponent's.
     *
     * @param prompts draws a prompt
     * @param promptsPerStep prompts per step
     * @param preference the preference model
     * @param random the random source
     * @return the mean win rate against the opponent
     */
    public double nashMdStep(Function<SplittableRandom, int[]> prompts, int promptsPerStep, Preference preference,
            SplittableRandom random) {
        check();
        int[][] rep = new int[promptsPerStep * samples][];
        for (int q = 0; q < promptsPerStep; q++) {
            int[] prompt = prompts.apply(random);
            for (int k = 0; k < samples; k++) {
                rep[q * samples + k] = prompt;
            }
        }
        int[][] mine = policy.sample(rep, maxTokens, end, 1.0, random);
        int[][] theirs = sampleMixture(rep, random);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<Float> adv = new ArrayList<>();
        double win = 0;
        for (int i = 0; i < rep.length; i++) {
            if (mine[i].length == 0) {
                continue;
            }
            double p = preference.matrix(rep[i], List.of(mine[i], theirs[i]))[0][1];
            win += p;
            ps.add(rep[i]);
            cs.add(mine[i]);
            adv.add((float) (p - 0.5));
        }
        if (cs.isEmpty()) {
            return 0.5;
        }
        float[] a = new float[adv.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = adv.get(i);
        }
        Var logp = policy.sequenceLogProbs(ps, cs);
        Var loss = logp.mul(Tensor.of(Shape.of(a.length, 1), a)).mean().neg();
        if (kl > 0) {
            Var d = Var.of(reference.sequenceLogProbs(ps, cs).value()).sub(logp);
            loss = loss.add(d.exp().sub(d).sub(1).mean().mul(kl));
        }
        loss.backward();
        optimizer.step(policy.parameters());
        return win / cs.size();
    }

    private void check() {
        if (end < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken) first");
        }
    }
}
