package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.SequenceScorer;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeCompute;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Process advantage verifiers, rewards for progress (Setlur et al., ICLR 2025,
 * <a href="https://arxiv.org/abs/2410.08146">arXiv:2410.08146</a>). A process reward model
 * that predicts "is the solution still right" (a Q-value of the base policy) gives RL little
 * signal: it rates a correct step that does not help as highly as one that does. A PAV rates
 * each step by its <em>advantage</em> under a separate <em>prover</em> policy {@code mu}: the
 * change in the prover's probability of eventually succeeding,
 * {@code A_mu(s, a) = V_mu(prefix + step) - V_mu(prefix)}. Provers that are complementary to the
 * base policy (even weaker ones) give the most useful signal, because they distinguish steps
 * the base policy would treat alike.
 *
 * <p>{@link #fit} estimates {@code V_mu} of step prefixes by Monte Carlo rollouts of the prover
 * and trains a {@link SequenceScorer} on them (like a PRM, on soft rollout targets);
 * {@link #progress} is then the difference of the scorer's values around each step. The step
 * rewards are used in two ways: as dense rewards added to the outcome advantage in online RL
 * ({@link #tokenAdvantages}) and, together with a Q-value PRM, as the beam score of step-level
 * search ({@link #beamScorer}).
 */
public final class ProgressReward {

    private final WindowLm prover;
    private final SequenceScorer value;
    private final int maxTokens;
    private final int end;
    private final int[] stepTokens;
    private int rollouts = 8;

    private ProgressReward(WindowLm prover, SequenceScorer value, int maxTokens, int end, int[] stepTokens) {
        this.prover = Objects.requireNonNull(prover);
        this.value = Objects.requireNonNull(value);
        this.maxTokens = maxTokens;
        this.end = end;
        this.stepTokens = stepTokens.clone();
    }

    /**
     * @param prover the prover policy {@code mu}
     * @param value the network that learns {@code V_mu} of prefixes
     * @param maxTokens the most tokens per solution
     * @param end the end token
     * @param stepTokens the tokens that end a step
     * @return the verifier
     */
    public static ProgressReward of(WindowLm prover, SequenceScorer value, int maxTokens, int end,
            int... stepTokens) {
        return new ProgressReward(prover, value, maxTokens, end, stepTokens);
    }

    /**
     * @param n prover rollouts per prefix when fitting
     * @return this verifier
     */
    public ProgressReward rollouts(int n) {
        this.rollouts = n;
        return this;
    }

    /** @return the value network */
    public SequenceScorer value() {
        return value;
    }

    private int[] stops() {
        int[] s = Arrays.copyOf(stepTokens, stepTokens.length + 1);
        s[stepTokens.length] = end;
        return s;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] o = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, o, a.length, b.length);
        return o;
    }

    /**
     * Fits {@code V_mu} on the step prefixes of given solutions (typically the base policy's).
     *
     * @param prompts the prompts
     * @param completions the solutions
     * @param reward the outcome reward (1 right, 0 wrong)
     * @param steps optimizer steps of the value network
     * @param random the random source
     * @return tokens the prover sampled
     */
    public long fit(List<int[]> prompts, List<int[]> completions, GrpoTrainer.Reward reward, int steps,
            SplittableRandom random) {
        List<int[]> seqs = new ArrayList<>();
        List<Float> targets = new ArrayList<>();
        long[] tokens = new long[1];
        for (int i = 0; i < prompts.size(); i++) {
            int[] p = prompts.get(i);
            int[] c = completions.get(i);
            int[] ends = TokenPolicyGradient.stepEnds(c, stops());
            for (int s = 0; s <= ends.length; s++) {
                int[] prefix = Arrays.copyOf(c, s == 0 ? 0 : ends[s - 1]);
                double v = MonteCarloValuePpo.rolloutValue(prover, p, prefix, reward, rollouts, maxTokens, end,
                        random, tokens);
                seqs.add(concat(p, prefix));
                targets.add((float) v);
            }
        }
        float[] y = new float[targets.size()];
        for (int i = 0; i < y.length; i++) {
            y[i] = targets.get(i);
        }
        value.fit(seqs, y, steps);
        return tokens[0];
    }

    /**
     * @param prompt the prompt
     * @param completion a (partial) solution
     * @return the progress {@code V(after) - V(before)} of each step
     */
    public double[] progress(int[] prompt, int[] completion) {
        int[] ends = TokenPolicyGradient.stepEnds(completion, stops());
        double[] out = new double[ends.length];
        double before = value.score(prompt);
        for (int s = 0; s < ends.length; s++) {
            double after = value.score(concat(prompt, Arrays.copyOf(completion, ends[s])));
            out[s] = after - before;
            before = after;
        }
        return out;
    }

    /**
     * Per-token advantages for online RL: each completion's outcome advantage plus
     * {@code alpha} times the progress of the step each token belongs to.
     *
     * @param prompts the prompts
     * @param completions the completions
     * @param outcomeAdvantages one outcome advantage per completion
     * @param alpha the weight of the progress reward
     * @return one advantage per token, in the order of {@link WindowLm#logProbs}
     */
    public float[] tokenAdvantages(List<int[]> prompts, List<int[]> completions, float[] outcomeAdvantages,
            float alpha) {
        List<Float> out = new ArrayList<>();
        for (int i = 0; i < completions.size(); i++) {
            int[] c = completions.get(i);
            double[] prog = progress(prompts.get(i), c);
            int[] ends = TokenPolicyGradient.stepEnds(c, stops());
            int start = 0;
            for (int s = 0; s < ends.length; s++) {
                for (int t = start; t < ends[s]; t++) {
                    out.add((float) (outcomeAdvantages[i] + alpha * prog[s]));
                }
                start = ends[s];
            }
        }
        float[] a = new float[out.size()];
        for (int i = 0; i < a.length; i++) {
            a[i] = out.get(i);
        }
        return a;
    }

    /**
     * The PAV beam score {@code Q(prefix) + alpha A_mu(last step)}.
     *
     * @param q the Q-value scorer (for example a process reward model's last-step score)
     * @param alpha the weight of the progress term
     * @return a scorer for {@code TestTimeCompute.stepBeamSearch}
     */
    public TestTimeCompute.Scorer beamScorer(TestTimeCompute.Scorer q, double alpha) {
        return (prompt, completion) -> {
            double[] prog = progress(prompt, completion);
            return q.score(prompt, completion) + alpha * (prog.length == 0 ? 0 : prog[prog.length - 1]);
        };
    }
}
