package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.BudgetForcing;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.LengthControlledReward;
import io.github.zzuegg.brainiac.experimental.llm.TokenSampler;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Length control learned with RL (LCPO, {@link LengthControlledReward}) against length
 * control forced at decode time ({@link BudgetForcing}, s1), on the bisection task of
 * {@link ThinkingBudget}: name a hidden {@code x} in {@code [0, 16)} after "thinking" by
 * bisection, where only four thoughts pin {@code x} down. The prompt is
 * {@code BOS x T}, where {@code T} states the thinking budget (0 to 4 thoughts) or is
 * {@code NONE}; a completion is the thoughts, the end-of-thinking token, the answer and a
 * stop token. The verifier checks every thought and that the answer follows from them, so
 * the best possible accuracy with {@code b} thoughts is {@code 2^b / 16}.
 *
 * <p>A {@link WindowLm} is first fine-tuned on traces that mostly stop early (as in
 * {@link ThinkingBudget}) with {@code T = NONE}, so it thinks about 1.3 thoughts on its own.
 * Budget forcing then imposes each budget on that model; LCPO instead trains it with GRPO and
 * the L1-Exact reward on prompts that state a random budget, and the trained model is asked
 * for each budget in its prompt and left to decode greedily.
 */
public final class LengthControlledThinking {

    /** The end-of-thinking token. */
    public static final int END = 16;
    private static final int BOS = 17;
    private static final int NONE = 18;
    private static final int BUDGET0 = 19;
    private static final int STOP = 24;
    private static final int VOCAB = 25;
    private static final int MAX_THOUGHTS = 4;
    private static final double[] LENGTHS = {0.35, 0.3, 0.2, 0.1, 0.05};

    /**
     * Accuracy and realized thinking length for each budget 0..4.
     *
     * @param unconstrainedAccuracy greedy accuracy of the fine-tuned model with no budget
     * @param unconstrainedLength its mean thinking length
     * @param forcedAccuracy budget forcing on the fine-tuned model, per budget
     * @param lcpoAccuracy the LCPO model asked for each budget, per budget
     * @param lcpoLength the LCPO model's mean realized thinking length, per budget
     */
    public record Result(double unconstrainedAccuracy, double unconstrainedLength, double[] forcedAccuracy,
            double[] lcpoAccuracy, double[] lcpoLength) {

        /** @return the mean absolute gap between the requested and the realized length under LCPO */
        public double lcpoLengthError() {
            double s = 0;
            for (int b = 0; b < lcpoLength.length; b++) {
                s += Math.abs(lcpoLength[b] - b);
            }
            return s / lcpoLength.length;
        }
    }

    private LengthControlledThinking() {
    }

    static int[] prompt(int x, int budgetToken) {
        return new int[] {BOS, x, budgetToken};
    }

    /** The thoughts, END, the answer the interval left implies, STOP. */
    static int[] completion(int x, int k) {
        int[] c = new int[k + 3];
        int lo = 0;
        int hi = 16;
        for (int i = 0; i < k; i++) {
            int m = (lo + hi) / 2;
            c[i] = m;
            if (x < m) {
                hi = m;
            } else {
                lo = m;
            }
        }
        c[k] = END;
        c[k + 1] = hi - lo == 1 ? lo : (lo + hi) / 2;
        c[k + 2] = STOP;
        return c;
    }

    /**
     * @param completion a completion
     * @return its thinking length: the tokens before the end-of-thinking token (all of them if none)
     */
    static int thinking(int[] completion) {
        for (int i = 0; i < completion.length; i++) {
            if (completion[i] == END) {
                return i;
            }
        }
        return completion.length;
    }

    /**
     * The verifier checks the reasoning, not only the answer: every thought must be the right
     * bisection midpoint, the answer must be the one the remaining interval implies, and it
     * must equal {@code x}. Copying {@code x} from the prompt without thinking is therefore
     * right only when the interval happens to imply it, as in {@link ThinkingBudget}.
     */
    static boolean correct(int[] prompt, int[] completion) {
        int k = thinking(completion);
        if (k > MAX_THOUGHTS || k + 2 > completion.length) {
            return false;
        }
        int[] expected = completion(prompt[1], k);
        for (int i = 0; i <= k + 1; i++) {
            if (completion[i] != expected[i]) {
                return false;
            }
        }
        return completion[k + 1] == prompt[1];
    }

    /**
     * Runs the comparison.
     *
     * @param sftSteps supervised steps on early-stopping traces
     * @param rlSteps LCPO steps
     * @param alpha the L1-Exact length penalty per thought
     * @param seed the seed
     * @return the measurements
     */
    public static Result run(int sftSteps, int rlSteps, float alpha, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        WindowLm lm = WindowLm.of(VOCAB, 8, 64, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < sftSteps; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int b = 0; b < 32; b++) {
                double u = r.nextDouble();
                int k = 0;
                while (k < LENGTHS.length - 1 && u > LENGTHS[k]) {
                    u -= LENGTHS[k];
                    k++;
                }
                int x = r.nextInt(16);
                ps.add(prompt(x, NONE));
                cs.add(completion(x, k));
            }
            lm.supervisedStep(ps, cs, opt);
        }
        int maxNew = MAX_THOUGHTS + 3;
        double free = 0;
        double freeLength = 0;
        for (int x = 0; x < 16; x++) {
            int[] c = lm.greedy(prompt(x, NONE), maxNew, STOP);
            free += correct(prompt(x, NONE), c) ? 1 : 0;
            freeLength += thinking(c);
        }
        double[] forced = new double[MAX_THOUGHTS + 1];
        for (int b = 0; b <= MAX_THOUGHTS; b++) {
            for (int x = 0; x < 16; x++) {
                BudgetForcing.Trace t = BudgetForcing.generate(lm, prompt(x, NONE), END, b, b, 1, TokenSampler.greedy(),
                        r);
                forced[b] += t.answer().length == 1 && t.answer()[0] == x ? 1 / 16.0 : 0;
            }
        }

        LengthControlledReward reward = new LengthControlledReward((p, c) -> correct(p, c) ? 1 : 0,
                p -> p[2] - BUDGET0, LengthControlledThinking::thinking, LengthControlledReward.Mode.EXACT, alpha, 0);
        GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(Optimizer.adam(3e-3f)).groupSize(8).promptsPerStep(16)
                .innerEpochs(2).klCoefficient(0.01f).completions(maxNew, STOP);
        for (int s = 0; s < rlSteps; s++) {
            grpo.step(rand -> prompt(rand.nextInt(16), BUDGET0 + rand.nextInt(MAX_THOUGHTS + 1)), reward, r);
        }
        double[] acc = new double[MAX_THOUGHTS + 1];
        double[] len = new double[MAX_THOUGHTS + 1];
        for (int b = 0; b <= MAX_THOUGHTS; b++) {
            for (int x = 0; x < 16; x++) {
                int[] p = prompt(x, BUDGET0 + b);
                int[] c = lm.greedy(p, maxNew, STOP);
                acc[b] += correct(p, c) ? 1 / 16.0 : 0;
                len[b] += thinking(c) / 16.0;
            }
        }
        return new Result(free / 16, freeLength / 16, forced, acc, len);
    }

    /**
     * @param values per-budget values
     * @return them, two decimals each
     */
    public static String format(double[] values) {
        return Arrays.toString(Arrays.stream(values).map(v -> Math.round(v * 100) / 100.0).toArray());
    }
}
