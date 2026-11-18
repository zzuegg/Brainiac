package io.github.zzuegg.brainiac.experimental.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntFunction;

/**
 * Adaptive analysts that overfit a reused holdout, to test {@link ReusableHoldout} mechanisms.
 * <ul>
 *   <li>{@link #boostingAttack}: Blum and Hardt's boosting attack
 *       (<a href="https://arxiv.org/abs/1502.04585">arXiv:1502.04585</a>): submit {@code k}
 *       random classifiers, keep those whose reported score looks better than chance (or, under a
 *       ladder, that moved the reported best), and submit their majority vote. On random labels
 *       the vote's holdout score climbs while its true accuracy stays at 1/2;</li>
 *   <li>{@link #greedySearch}: an honest search that queries candidates one by one and keeps
 *       the one with the best reported score.</li>
 * </ul>
 */
public final class AdaptiveAnalyst {

    private AdaptiveAnalyst() {
    }

    /**
     * The outcome of an adaptive analysis.
     *
     * @param reported the score the mechanism reported for the final submission
     * @param holdout the final submission's raw holdout score
     * @param truth its score on fresh data
     * @param kept submissions kept along the way
     */
    public record Outcome(double reported, double holdout, double truth, int kept) {
    }

    /** A classifier evaluated on the three example sets. */
    public interface Submission {
        /**
         * @param set 0 train, 1 holdout, 2 fresh
         * @param index the example
         * @return its predicted label (0 or 1)
         */
        int predict(int set, int index);
    }

    private static double[] correct(Submission s, int set, int[] labels) {
        double[] c = new double[labels.length];
        for (int i = 0; i < c.length; i++) {
            c[i] = s.predict(set, i) == labels[i] ? 1 : 0;
        }
        return c;
    }

    private static double mean(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    /**
     * The boosting attack.
     *
     * @param mechanism the holdout mechanism
     * @param trainLabels training labels
     * @param holdoutLabels holdout labels
     * @param freshLabels fresh labels (the truth)
     * @param k random submissions
     * @param randomClassifier the {@code i}-th random classifier
     * @return the outcome
     */
    public static Outcome boostingAttack(ReusableHoldout mechanism, int[] trainLabels, int[] holdoutLabels,
            int[] freshLabels, int k, IntFunction<Submission> randomClassifier) {
        List<Submission> kept = new ArrayList<>();
        double previous = Double.NEGATIVE_INFINITY;
        boolean ladder = mechanism.kind() == ReusableHoldout.Kind.LADDER
                || mechanism.kind() == ReusableHoldout.Kind.PARAMETER_FREE_LADDER;
        for (int i = 0; i < k; i++) {
            Submission s = randomClassifier.apply(i);
            double reported = mechanism.query(correct(s, 1, holdoutLabels), correct(s, 0, trainLabels));
            boolean keep = ladder ? reported > previous : reported > 0.5;
            if (keep) {
                kept.add(s);
            }
            previous = Math.max(previous, reported);
        }
        Submission vote = (set, index) -> {
            int ones = 0;
            for (Submission s : kept) {
                ones += s.predict(set, index);
            }
            return 2 * ones > kept.size() ? 1 : 0;
        };
        double[] h = correct(vote, 1, holdoutLabels);
        double reported = mechanism.query(h, correct(vote, 0, trainLabels));
        return new Outcome(reported, mean(h), mean(correct(vote, 2, freshLabels)), kept.size());
    }

    /**
     * A greedy search: query every candidate, keep the best reported one.
     *
     * @param mechanism the holdout mechanism
     * @param trainScores per-candidate per-example training scores
     * @param holdoutScores per-candidate per-example holdout scores
     * @param freshScores per-candidate per-example fresh scores
     * @return the outcome of the chosen candidate
     */
    public static Outcome greedySearch(ReusableHoldout mechanism, List<double[]> trainScores,
            List<double[]> holdoutScores, List<double[]> freshScores) {
        int best = -1;
        double bestReported = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < holdoutScores.size(); c++) {
            double r = mechanism.query(holdoutScores.get(c), trainScores.get(c));
            if (r > bestReported) {
                bestReported = r;
                best = c;
            }
        }
        return new Outcome(bestReported, mean(holdoutScores.get(best)), mean(freshScores.get(best)), 1);
    }
}
