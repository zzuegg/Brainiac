package io.github.zzuegg.brainiac.experimental.align;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * XPO, exploratory preference optimization (Xie et al., ICLR 2025,
 * <a href="https://arxiv.org/abs/2405.21046">arXiv:2405.21046</a>): online DPO with one extra
 * term. Each round samples a response {@code y} from the current policy and a response
 * {@code y~} from the reference, asks a preference oracle which is better, adds the labelled
 * pair to a growing dataset and refits the policy on it with
 * {@code DPO(pairs) + alpha mean log pi(y~)}. Minimizing the log-likelihood of the
 * reference's samples is an optimism bonus: it pushes probability away from where the reference
 * already is, towards responses the reference rarely produces, which gives online preference
 * learning the sample-efficiency guarantees of optimistic exploration in RL. With
 * {@code alpha = 0} and both responses from the policy ({@link #onlineDpo}) it is plain online
 * (iterative) DPO.
 */
public final class ExploratoryPreferenceOptimizer {

    /** Labels a pair: the preference oracle (a human, or a gold reward in experiments). */
    @FunctionalInterface
    public interface Oracle {
        /**
         * @param prompt the prompt
         * @param a one response
         * @param b another
         * @param random the oracle's own randomness
         * @return whether {@code a} is preferred
         */
        boolean prefersFirst(int[] prompt, int[] a, int[] b, SplittableRandom random);
    }

    private final WindowLm policy;
    private final WindowLm reference;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float alpha = 0.1f;
    private float beta = 0.5f;
    private boolean fromReference = true;
    private int maxTokens = 4;
    private int end = -1;
    private int fitSteps = 4;
    private int batch = 32;
    private long queries;
    private final List<DpoTrainer.Preference> data = new ArrayList<>();
    private final List<int[]> explorePrompts = new ArrayList<>();
    private final List<int[]> exploreResponses = new ArrayList<>();

    private ExploratoryPreferenceOptimizer(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = policy.copy();
    }

    /**
     * @param policy the model to train; a frozen copy becomes the reference
     * @return the optimizer, XPO by default
     */
    public static ExploratoryPreferenceOptimizer of(WindowLm policy) {
        return new ExploratoryPreferenceOptimizer(policy);
    }

    /**
     * Plain online DPO: no optimism bonus, both responses from the current policy.
     *
     * @return this optimizer
     */
    public ExploratoryPreferenceOptimizer onlineDpo() {
        this.alpha = 0;
        this.fromReference = false;
        return this;
    }

    /**
     * @param a the optimism weight
     * @param b the DPO inverse temperature
     * @return this optimizer
     */
    public ExploratoryPreferenceOptimizer strength(float a, float b) {
        this.alpha = a;
        this.beta = b;
        return this;
    }

    /**
     * @param o the optimizer
     * @return this optimizer
     */
    public ExploratoryPreferenceOptimizer optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param max the most tokens per response
     * @param endToken the end token
     * @return this optimizer
     */
    public ExploratoryPreferenceOptimizer completions(int max, int endToken) {
        this.maxTokens = max;
        this.end = endToken;
        return this;
    }

    /**
     * @param steps gradient steps on the dataset per round
     * @param pairs pairs per gradient step
     * @return this optimizer
     */
    public ExploratoryPreferenceOptimizer fitting(int steps, int pairs) {
        this.fitSteps = steps;
        this.batch = pairs;
        return this;
    }

    /** @return preference queries made so far */
    public long queries() {
        return queries;
    }

    /**
     * One round: collect and label {@code pairs} new pairs, then refit.
     *
     * @param prompts draws a prompt
     * @param pairs new pairs this round
     * @param oracle the preference oracle
     * @param random the random source
     */
    public void round(Function<SplittableRandom, int[]> prompts, int pairs, Oracle oracle, SplittableRandom random) {
        if (end < 0) {
            throw new IllegalStateException("call completions(maxTokens, endToken) first");
        }
        int[][] ps = new int[pairs][];
        for (int i = 0; i < pairs; i++) {
            ps[i] = prompts.apply(random);
        }
        int[][] mine = policy.sample(ps, maxTokens, end, 1.0, random);
        int[][] other = (fromReference ? reference : policy).sample(ps, maxTokens, end, 1.0, random);
        for (int i = 0; i < pairs; i++) {
            if (other[i].length > 0) {
                explorePrompts.add(ps[i]);
                exploreResponses.add(other[i]);
            }
            if (mine[i].length == 0 || other[i].length == 0 || Arrays.equals(mine[i], other[i])) {
                continue;
            }
            queries++;
            boolean first = oracle.prefersFirst(ps[i], mine[i], other[i], random);
            data.add(new DpoTrainer.Preference(ps[i], first ? mine[i] : other[i], first ? other[i] : mine[i]));
        }
        if (data.isEmpty()) {
            return;
        }
        for (int s = 0; s < fitSteps; s++) {
            List<DpoTrainer.Preference> b = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                b.add(data.get(random.nextInt(data.size())));
            }
            LinkDpoLosses.Batch lb = LinkDpoLosses.batch(policy, reference, b, false);
            Var loss = PreferenceLosses.dpo(lb.policyChosen(), lb.policyRejected(), lb.refChosen(), lb.refRejected(),
                    beta);
            if (alpha > 0 && !exploreResponses.isEmpty()) {
                List<int[]> ep = new ArrayList<>();
                List<int[]> er = new ArrayList<>();
                for (int i = 0; i < batch; i++) {
                    int j = random.nextInt(exploreResponses.size());
                    ep.add(explorePrompts.get(j));
                    er.add(exploreResponses.get(j));
                }
                loss = loss.add(policy.sequenceLogProbs(ep, er).mean().mul(alpha));
            }
            loss.backward();
            optimizer.step(policy.parameters());
        }
    }
}
