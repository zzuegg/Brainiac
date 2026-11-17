package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.BudgetForcing;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.TokenSampler;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Budget forcing on a tiny reasoner that thinks too little. The task: name a hidden number
 * {@code x} in {@code [0, 16)} given as the prompt, after "thinking" by bisection; each thought
 * is the next midpoint, and the answer after the end-of-thinking token is the middle of the
 * interval left, which is {@code x} only after four thoughts. The training traces stop early
 * most of the time (k thoughts with probabilities 0.35, 0.3, 0.2, 0.1, 0.05 for k = 0..4), so
 * left alone the model ends its thinking early and guesses. {@link BudgetForcing} then fixes
 * the thinking length at decode time; accuracy is measured against the budget and against
 * unconstrained greedy generation.
 */
public final class ThinkingBudget {

    /** The end-of-thinking token. */
    public static final int END = 16;
    private static final int BOS = 17;
    private static final int PAD = 18;
    private static final int VOCAB = 19;
    private static final double[] LENGTHS = {0.35, 0.3, 0.2, 0.1, 0.05};

    /**
     * The measurements.
     *
     * @param unconstrained greedy accuracy with no budget
     * @param unconstrainedThinking the mean thinking length with no budget
     * @param byBudget accuracy with thinking forced to exactly each budget
     */
    public record Result(double unconstrained, double unconstrainedThinking, Map<Integer, Double> byBudget) {
    }

    private ThinkingBudget() {
    }

    /**
     * A training trace: {@code BOS x m1..mk END answer PAD...}, eight tokens.
     *
     * @param x the hidden number
     * @param k the thoughts
     * @return the trace
     */
    static int[] trace(int x, int k) {
        int[] t = new int[8];
        java.util.Arrays.fill(t, PAD);
        t[0] = BOS;
        t[1] = x;
        int lo = 0;
        int hi = 16;
        for (int i = 0; i < k; i++) {
            int m = (lo + hi) / 2;
            t[2 + i] = m;
            if (x < m) {
                hi = m;
            } else {
                lo = m;
            }
        }
        t[2 + k] = END;
        t[3 + k] = answer(lo, hi);
        return t;
    }

    private static int answer(int lo, int hi) {
        return hi - lo == 1 ? lo : (lo + hi) / 2;
    }

    /**
     * Trains the reasoner and sweeps the budget.
     *
     * @param steps training steps
     * @param seed the seed
     * @return the measurements
     */
    public static Result run(int steps, long seed) {
        HookedDecoder model = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB), seed);
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[][] batch = new int[32][];
            for (int b = 0; b < batch.length; b++) {
                double u = r.nextDouble();
                int k = 0;
                while (k < LENGTHS.length - 1 && u > LENGTHS[k]) {
                    u -= LENGTHS[k];
                    k++;
                }
                batch[b] = trace(r.nextInt(16), k);
            }
            Var loss = model.loss(batch, true);
            loss.backward();
            opt.step(model.parameters());
            model.zeroGrad();
        }
        int free = 0;
        int thought = 0;
        for (int x = 0; x < 16; x++) {
            BudgetForcing.Trace t = BudgetForcing.generate(model, new int[] {BOS, x}, END, 0, 8, 1,
                    TokenSampler.greedy(), r);
            free += t.answer().length == 1 && t.answer()[0] == x ? 1 : 0;
            thought += t.thinking().length;
        }
        Map<Integer, Double> byBudget = new LinkedHashMap<>();
        for (int budget = 0; budget <= 4; budget++) {
            int right = 0;
            for (int x = 0; x < 16; x++) {
                BudgetForcing.Trace t = BudgetForcing.generate(model, new int[] {BOS, x}, END, budget, budget, 1,
                        TokenSampler.greedy(), r);
                right += t.answer().length == 1 && t.answer()[0] == x ? 1 : 0;
            }
            byBudget.put(budget, right / 16.0);
        }
        return new Result(free / 16.0, thought / 16.0, byBudget);
    }
}
