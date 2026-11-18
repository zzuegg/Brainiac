package io.github.zzuegg.brainiac.experimental.eval;

import java.util.SplittableRandom;

/**
 * Holdout sets that survive adaptive reuse. Every score an analyst reads off a test set leaks
 * information about it; enough adaptive queries overfit the test set itself (a leaderboard can be
 * climbed by boosting random submissions). Each mechanism answers "what is this submission's
 * score?" with a value that stays close to the true (population) score.
 * <ul>
 *   <li>{@link #plain}: the reused holdout, the score as is;</li>
 *   <li>{@link #ladder}: the Ladder (Blum and Hardt, ICML 2015,
 *       <a href="https://arxiv.org/abs/1502.04585">arXiv:1502.04585</a>): report a new score
 *       only if it beats the best so far by a margin η, rounded to a multiple of η; otherwise
 *       repeat the best;</li>
 *   <li>{@link #parameterFreeLadder}: the same with the margin set per submission to the
 *       standard error of the paired score difference against the current best;</li>
 *   <li>{@link #thresholdout}: Thresholdout (Dwork et al., Science 2015,
 *       <a href="https://arxiv.org/abs/1506.02629">arXiv:1506.02629</a>): answer with the
 *       training-set score unless it differs from the holdout score by more than a noisy
 *       threshold; then answer with the holdout score plus Laplace noise and spend budget.</li>
 * </ul>
 * Scores are higher-is-better per-example values (for example 0/1 correctness), given as the
 * per-example vectors so the mechanisms can compute means and paired differences.
 */
public final class ReusableHoldout {

    /** The mechanism. */
    public enum Kind {
        /** The plain reused holdout. */
        PLAIN,
        /** The Ladder with a fixed margin. */
        LADDER,
        /** The parameter-free Ladder. */
        PARAMETER_FREE_LADDER,
        /** Thresholdout. */
        THRESHOLDOUT
    }

    private final Kind kind;
    private final double eta;
    private final double threshold;
    private final double sigma;
    private final SplittableRandom random;
    private double best = Double.NEGATIVE_INFINITY;
    private double[] bestScores;
    private int budget;
    private double noisyThreshold;
    private int queries;

    private ReusableHoldout(Kind kind, double eta, double threshold, double sigma, int budget, long seed) {
        this.kind = kind;
        this.eta = eta;
        this.threshold = threshold;
        this.sigma = sigma;
        this.budget = budget;
        this.random = new SplittableRandom(seed);
        this.noisyThreshold = threshold + laplace(2 * sigma);
    }

    /**
     * The plain reused holdout.
     *
     * @return the mechanism
     */
    public static ReusableHoldout plain() {
        return new ReusableHoldout(Kind.PLAIN, 0, 0, 0, Integer.MAX_VALUE, 0);
    }

    /**
     * The Ladder.
     *
     * @param eta the margin and rounding step, for example 0.01
     * @return the mechanism
     */
    public static ReusableHoldout ladder(double eta) {
        return new ReusableHoldout(Kind.LADDER, eta, 0, 0, Integer.MAX_VALUE, 0);
    }

    /**
     * The parameter-free Ladder.
     *
     * @return the mechanism
     */
    public static ReusableHoldout parameterFreeLadder() {
        return new ReusableHoldout(Kind.PARAMETER_FREE_LADDER, 0, 0, 0, Integer.MAX_VALUE, 0);
    }

    /**
     * Thresholdout.
     *
     * @param threshold the threshold T, for example 0.04
     * @param sigma the Laplace scale σ, for example 0.01
     * @param budget answers from the holdout before it is spent
     * @param seed the noise seed
     * @return the mechanism
     */
    public static ReusableHoldout thresholdout(double threshold, double sigma, int budget, long seed) {
        return new ReusableHoldout(Kind.THRESHOLDOUT, 0, threshold, sigma, budget, seed);
    }

    /** @return the mechanism's kind */
    public Kind kind() {
        return kind;
    }

    /** @return queries answered so far */
    public int queries() {
        return queries;
    }

    /** @return holdout answers left (Thresholdout) */
    public int budget() {
        return budget;
    }

    private double laplace(double scale) {
        double u = random.nextDouble() - 0.5;
        return -scale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
    }

    private static double mean(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    /**
     * Answers one query.
     *
     * @param holdoutScores the submission's per-example scores on the holdout
     * @param trainScores its per-example scores on the training set (Thresholdout only; may be
     *     null for the others)
     * @return the reported score
     */
    public double query(double[] holdoutScores, double[] trainScores) {
        queries++;
        double h = mean(holdoutScores);
        return switch (kind) {
            case PLAIN -> h;
            case LADDER -> {
                if (h > best + eta) {
                    best = Math.round(h / eta) * eta;
                }
                yield best;
            }
            case PARAMETER_FREE_LADDER -> {
                if (bestScores == null) {
                    bestScores = holdoutScores.clone();
                    best = h;
                    yield best;
                }
                double[] d = new double[holdoutScores.length];
                for (int i = 0; i < d.length; i++) {
                    d[i] = holdoutScores[i] - bestScores[i];
                }
                double md = mean(d);
                double s = 0;
                for (double v : d) {
                    s += (v - md) * (v - md);
                }
                double se = Math.sqrt(s / Math.max(1, d.length - 1)) / Math.sqrt(d.length);
                if (md > se) {
                    bestScores = holdoutScores.clone();
                    best = h;
                }
                yield best;
            }
            case THRESHOLDOUT -> {
                double t = mean(trainScores);
                if (budget <= 0) {
                    yield t;
                }
                if (Math.abs(t - h) > noisyThreshold + laplace(4 * sigma)) {
                    budget--;
                    noisyThreshold = threshold + laplace(2 * sigma);
                    yield h + laplace(sigma);
                }
                yield t;
            }
        };
    }

    @Override
    public String toString() {
        return "ReusableHoldout(" + kind + ", " + queries + " queries)";
    }
}
