package io.github.zzuegg.brainiac.experimental.train;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Privacy auditing with one training run (Steinke, Nasr and Jagielski 2023, "Privacy
 * Auditing with One (1) Training Run", <a href="https://arxiv.org/abs/2305.08846">arXiv:2305.08846</a>).
 *
 * <p>Each of {@code m} canaries is put into the training data independently with
 * probability one half. After training, an attacker scores every canary (higher means
 * "more likely trained on"), guesses "in" for the {@code k+} highest scores and "out" for
 * the {@code k-} lowest, and abstains on the rest. Under {@code ε}-DP each guess is right with
 * probability at most {@code e^ε / (1 + e^ε)}, even given the other guesses, so many right
 * guesses refute small {@code ε}. The lower bound is the largest {@code ε} for which the
 * observed number of right guesses still has probability below {@code β} under the
 * binomial tail. This is the paper's pure-DP bound ({@code δ = 0}); it is slightly too
 * high for an {@code (ε, δ)} guarantee with tiny {@code δ}.
 */
public final class OneRunAudit {

    /**
     * The audit's outcome.
     *
     * @param guesses guesses made
     * @param correct guesses that were right
     * @param epsilonLowerBound the {@code ε} the run provably exceeds with confidence {@code 1 - β}
     */
    public record Result(int guesses, int correct, double epsilonLowerBound) {
    }

    private OneRunAudit() {
    }

    /**
     * Audits one run.
     *
     * @param scores the attacker's score per canary, higher meaning "in"
     * @param included whether each canary was trained on
     * @param positives guesses of "in", for the highest scores
     * @param negatives guesses of "out", for the lowest scores
     * @param beta the confidence level, such as 0.05
     * @return the outcome
     */
    public static Result audit(double[] scores, boolean[] included, int positives, int negatives, double beta) {
        if (scores.length != included.length || positives + negatives > scores.length) {
            throw new IllegalArgumentException("bad audit sizes");
        }
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.comparingDouble(i -> -scores[i]));
        int correct = 0;
        for (int i = 0; i < positives; i++) {
            correct += included[order[i]] ? 1 : 0;
        }
        for (int i = 0; i < negatives; i++) {
            correct += included[order[order.length - 1 - i]] ? 0 : 1;
        }
        int guesses = positives + negatives;
        return new Result(guesses, correct, epsilonLowerBound(guesses, correct, beta));
    }

    /**
     * The largest {@code ε} under which {@code correct} or more right answers out of
     * {@code guesses} have probability below {@code beta}.
     *
     * @param guesses guesses made
     * @param correct guesses that were right
     * @param beta the confidence level
     * @return the lower bound, 0 if no {@code ε} is refuted
     */
    public static double epsilonLowerBound(int guesses, int correct, double beta) {
        if (tail(guesses, correct, 0.5) >= beta) {
            return 0;
        }
        double lo = 0;
        double hi = 50;
        for (int i = 0; i < 100; i++) {
            double mid = (lo + hi) / 2;
            double p = 1 / (1 + Math.exp(-mid));
            if (tail(guesses, correct, p) < beta) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    /** {@code P[Binomial(n, p) >= k]}, summed in log space. */
    static double tail(int n, int k, double p) {
        if (k <= 0) {
            return 1;
        }
        double logBinom = 0;
        double sum = 0;
        for (int i = 0; i <= n; i++) {
            if (i > 0) {
                logBinom += Math.log(n - i + 1) - Math.log(i);
            }
            if (i >= k) {
                sum += Math.exp(logBinom + i * Math.log(p) + (n - i) * Math.log1p(-p));
            }
        }
        return Math.min(1, sum);
    }
}
