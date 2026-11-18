package io.github.zzuegg.brainiac.experimental.eval;

import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Bias probes and mitigations for pairwise LM judges (Zheng et al., "Judging LLM-as-a-Judge",
 * NeurIPS 2023, <a href="https://arxiv.org/abs/2306.05685">arXiv:2306.05685</a>; Dubois et al.,
 * "Length-Controlled AlpacaEval", 2024, <a href="https://arxiv.org/abs/2404.04475">arXiv:2404.04475</a>;
 * Panickssery et al., "LLM Evaluators Recognize and Favor Their Own Generations", NeurIPS 2024,
 * <a href="https://arxiv.org/abs/2404.13076">arXiv:2404.13076</a>).
 * <ul>
 *   <li>{@link #positionBias}: how much the judge favours the answer shown first, from judging
 *       every pair in both orders;</li>
 *   <li>{@link #probe}: the verdict change when one answer is transformed in a way that should
 *       not matter (padding without content, a different source style), swap-averaged so the
 *       position effect cancels;</li>
 *   <li>{@link #swapAveraged}: the order-debiased verdict {@code (P(a≻b | a first) + 1 - P(b≻a | b first)) / 2};</li>
 *   <li>{@link #lengthControlled}: length control as in length-controlled AlpacaEval: a
 *       logistic GLM of the judge's preference for system {@code s} over the baseline on a
 *       per-system intercept plus a shared length term {@code φ · tanh(Δlen / sd)}, reported as
 *       the win rate with the length term set to zero, {@code σ(θ_s)}.</li>
 * </ul>
 */
public final class JudgeAudit {

    private JudgeAudit() {
    }

    /** A pairwise judge. */
    @FunctionalInterface
    public interface PairJudge {
        /**
         * @param first the answer shown first
         * @param second the answer shown second
         * @return the probability that the first answer is better
         */
        double prefer(int[] first, int[] second);
    }

    /**
     * The order-debiased verdict.
     *
     * @param judge the judge
     * @param a one answer
     * @param b the other
     * @return the swap-averaged probability that {@code a} is better
     */
    public static double swapAveraged(PairJudge judge, int[] a, int[] b) {
        return 0.5 * (judge.prefer(a, b) + 1 - judge.prefer(b, a));
    }

    /**
     * The position bias: the mean probability that the first-shown answer wins, minus 1/2,
     * over pairs judged in both orders.
     *
     * @param judge the judge
     * @param pairs answer pairs
     * @return the bias in [-1/2, 1/2]
     */
    public static double positionBias(PairJudge judge, List<int[][]> pairs) {
        double s = 0;
        for (int[][] p : pairs) {
            s += 0.5 * (judge.prefer(p[0], p[1]) + judge.prefer(p[1], p[0]));
        }
        return s / pairs.size() - 0.5;
    }

    /**
     * The swap-averaged verdict change when the first answer of every pair is transformed.
     *
     * @param judge the judge
     * @param pairs answer pairs
     * @param transform a content-preserving change (padding, a style marker)
     * @return the mean change of {@code P(a ≻ b)}
     */
    public static double probe(PairJudge judge, List<int[][]> pairs, UnaryOperator<int[]> transform) {
        double s = 0;
        for (int[][] p : pairs) {
            s += swapAveraged(judge, transform.apply(p[0]), p[1]) - swapAveraged(judge, p[0], p[1]);
        }
        return s / pairs.size();
    }

    /**
     * The share of pairs whose verdict flips (crosses 1/2) when the order is swapped.
     *
     * @param judge the judge
     * @param pairs answer pairs
     * @return the inconsistency rate
     */
    public static double inconsistency(PairJudge judge, List<int[][]> pairs) {
        int n = 0;
        for (int[][] p : pairs) {
            boolean ab = judge.prefer(p[0], p[1]) > 0.5;
            boolean ba = judge.prefer(p[1], p[0]) > 0.5;
            n += ab == ba ? 1 : 0;
        }
        return n / (double) pairs.size();
    }

    /**
     * Raw win rates: the mean verdict per system.
     *
     * @param verdicts {@code P(system's answer ≻ baseline)} per comparison
     * @param system the system of each comparison
     * @param systems the number of systems
     * @return the win rate of each system
     */
    public static double[] winRates(double[] verdicts, int[] system, int systems) {
        double[] s = new double[systems];
        int[] n = new int[systems];
        for (int i = 0; i < verdicts.length; i++) {
            s[system[i]] += verdicts[i];
            n[system[i]]++;
        }
        for (int k = 0; k < systems; k++) {
            s[k] /= Math.max(1, n[k]);
        }
        return s;
    }

    /**
     * Length-controlled win rates: fits {@code logit P = θ_s + φ tanh(Δlen / sd(Δlen))} to the
     * (soft) verdicts by gradient descent on the cross-entropy, then returns {@code σ(θ_s)}.
     *
     * @param verdicts {@code P(system's answer ≻ baseline)} per comparison
     * @param system the system of each comparison
     * @param lengthDifference system answer length minus baseline answer length
     * @param systems the number of systems
     * @return {@code [systems + 1]}: the length-controlled win rate of each system, then φ
     */
    public static double[] lengthControlled(double[] verdicts, int[] system, double[] lengthDifference, int systems) {
        double sd = 0;
        double mean = 0;
        for (double d : lengthDifference) {
            mean += d / lengthDifference.length;
        }
        for (double d : lengthDifference) {
            sd += (d - mean) * (d - mean) / lengthDifference.length;
        }
        sd = Math.max(1e-9, Math.sqrt(sd));
        double[] feature = new double[verdicts.length];
        for (int i = 0; i < feature.length; i++) {
            feature[i] = Math.tanh(lengthDifference[i] / sd);
        }
        double[] theta = new double[systems];
        double phi = 0;
        int[] count = new int[systems];
        for (int s : system) {
            count[s]++;
        }
        for (int it = 0; it < 3000; it++) {
            double[] gTheta = new double[systems];
            double gPhi = 0;
            for (int i = 0; i < verdicts.length; i++) {
                double p = 1 / (1 + Math.exp(-(theta[system[i]] + phi * feature[i])));
                double e = p - verdicts[i];
                gTheta[system[i]] += e / Math.max(1, count[system[i]]);
                gPhi += e * feature[i] / verdicts.length;
            }
            for (int k = 0; k < systems; k++) {
                theta[k] -= 2.0 * gTheta[k];
            }
            phi -= 2.0 * gPhi;
        }
        double[] out = new double[systems + 1];
        for (int k = 0; k < systems; k++) {
            out[k] = 1 / (1 + Math.exp(-theta[k]));
        }
        out[systems] = phi;
        return out;
    }
}
