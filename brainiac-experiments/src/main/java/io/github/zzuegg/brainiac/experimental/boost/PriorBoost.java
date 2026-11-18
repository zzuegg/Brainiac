package io.github.zzuegg.brainiac.experimental.boost;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * PFN-Boost (Jayawardhana et al., "Transformers Boost the Performance of Decision Trees on
 * Tabular Data across Sample Sizes", 2025, <a href="https://arxiv.org/abs/2502.02672">
 * arXiv:2502.02672</a>): gradient-boosted trees that start from a prior model's scores instead of
 * a constant. The prior is any in-context scorer, such as a prior-fitted network (TabPFN) or an
 * LLM's label likelihoods; its per-class log-probabilities, multiplied by a scale {@code s},
 * become the boosting's starting margins and the trees fit what the prior gets wrong. With
 * {@code s = 0} this is plain boosting; with large {@code s} it follows the prior; the scale is
 * chosen on a validation split. Training rows get <em>out-of-fold</em> prior scores, so the trees
 * never see margins computed with the row's own label in the context.
 *
 * <p>Binary tasks boost one {@link LightBoost} on the log-odds margin; multiclass tasks boost one
 * per class on that class's log-odds (one-vs-rest) and renormalise.
 */
public final class PriorBoost {

    /** A prior model scored in context: labelled rows in, class probabilities for new rows out. */
    @FunctionalInterface
    public interface Scorer {
        /**
         * @param trainX context rows
         * @param trainY their labels
         * @param testX rows to score
         * @return {@code [testRows][classes]} probabilities
         */
        double[][] probabilities(double[][] trainX, int[] trainY, double[][] testX);
    }

    /**
     * Settings.
     *
     * @param scales candidate prior scales {@code s}
     * @param folds out-of-fold splits for the training rows' prior scores
     * @param validation the share of training rows held out to choose {@code s}
     * @param boosting the booster settings
     * @param seed seeds the splits
     */
    public record Settings(double[] scales, int folds, double validation, LightBoost.Settings boosting, long seed) {

        /**
         * @param boosting the booster settings
         * @return scales {@code {0, 0.25, 0.5, 1, 2}}, 4 folds and a 25% validation split
         */
        public static Settings defaults(LightBoost.Settings boosting) {
            return new Settings(new double[] {0, 0.25, 0.5, 1, 2}, 4, 0.25, boosting, 1);
        }
    }

    private final Scorer scorer;
    private final double[][] trainX;
    private final int[] trainY;
    private final int classes;
    private final double scale;
    private final LightBoost[] boosters;

    private PriorBoost(Scorer scorer, double[][] trainX, int[] trainY, int classes, double scale,
            LightBoost[] boosters) {
        this.scorer = scorer;
        this.trainX = trainX;
        this.trainY = trainY;
        this.classes = classes;
        this.scale = scale;
        this.boosters = boosters;
    }

    /**
     * Chooses the scale on a validation split, then refits on every training row.
     *
     * @param x training rows
     * @param y labels in {@code [0, classes)}
     * @param classes the number of classes
     * @param scorer the prior
     * @param s the settings
     * @return the fitted model
     */
    public static PriorBoost fit(double[][] x, int[] y, int classes, Scorer scorer, Settings s) {
        int n = y.length;
        int[] perm = permutation(n, new SplittableRandom(s.seed()));
        int held = Math.max(1, (int) Math.round(n * s.validation()));
        int[] fitRows = Arrays.copyOfRange(perm, held, n);
        int[] valRows = Arrays.copyOfRange(perm, 0, held);
        double[][] fx = rows(x, fitRows);
        int[] fy = labels(y, fitRows);
        double[][] vx = rows(x, valRows);
        int[] vy = labels(y, valRows);
        double[][] fitPrior = outOfFold(fx, fy, classes, scorer, s.folds(), s.seed());
        double[][] valPrior = scorer.probabilities(fx, fy, vx);
        double bestScale = s.scales()[0];
        double bestAcc = -1;
        for (double scale : s.scales()) {
            LightBoost[] b = boost(fx, fy, classes, fitPrior, scale, s.boosting());
            double acc = accuracy(predict(b, vx, valPrior, scale, classes), vy);
            if (acc > bestAcc + 1e-12) {
                bestAcc = acc;
                bestScale = scale;
            }
        }
        double[][] prior = outOfFold(x, y, classes, scorer, s.folds(), s.seed());
        LightBoost[] boosters = boost(x, y, classes, prior, bestScale, s.boosting());
        return new PriorBoost(scorer, x, y, classes, bestScale, boosters);
    }

    /**
     * @param testX rows to classify
     * @return {@code [rows][classes]} probabilities
     */
    public double[][] predictProbability(double[][] testX) {
        return predict(boosters, testX, scorer.probabilities(trainX, trainY, testX), scale, classes);
    }

    /** @return the chosen prior scale {@code s} */
    public double scale() {
        return scale;
    }

    private static double logit(double p) {
        double q = Math.clamp(p, 1e-6, 1 - 1e-6);
        return Math.log(q / (1 - q));
    }

    private static LightBoost[] boost(double[][] x, int[] y, int classes, double[][] prior, double scale,
            LightBoost.Settings settings) {
        int heads = classes == 2 ? 1 : classes;
        LightBoost[] out = new LightBoost[heads];
        for (int c = 0; c < heads; c++) {
            int cls = classes == 2 ? 1 : c;
            double[] target = new double[y.length];
            double[] margin = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                target[i] = y[i] == cls ? 1 : 0;
                margin[i] = scale * logit(prior[i][cls]);
            }
            // s = 0 is plain boosting from the class prior (the loss's constant start)
            out[c] = LightBoost.fit(x, target, scale == 0 ? null : margin, BoostLosses.logistic(), settings);
        }
        return out;
    }

    private static double[][] predict(LightBoost[] boosters, double[][] x, double[][] prior, double scale,
            int classes) {
        double[][] out = new double[x.length][classes];
        for (int i = 0; i < x.length; i++) {
            if (classes == 2) {
                double raw = scale == 0 ? boosters[0].raw(x[i]) : boosters[0].raw(x[i], scale * logit(prior[i][1]));
                double p = 1 / (1 + Math.exp(-raw));
                out[i][1] = p;
                out[i][0] = 1 - p;
                continue;
            }
            double sum = 0;
            for (int c = 0; c < classes; c++) {
                double raw = scale == 0 ? boosters[c].raw(x[i]) : boosters[c].raw(x[i], scale * logit(prior[i][c]));
                out[i][c] = 1 / (1 + Math.exp(-raw));
                sum += out[i][c];
            }
            for (int c = 0; c < classes; c++) {
                out[i][c] /= sum;
            }
        }
        return out;
    }

    /**
     * Out-of-fold prior probabilities for the training rows.
     *
     * @param x rows
     * @param y labels
     * @param classes classes
     * @param scorer the prior
     * @param folds folds
     * @param seed the fold seed
     * @return {@code [rows][classes]}
     */
    public static double[][] outOfFold(double[][] x, int[] y, int classes, Scorer scorer, int folds, long seed) {
        int n = y.length;
        int[] perm = permutation(n, new SplittableRandom(seed + 31));
        double[][] out = new double[n][];
        for (int f = 0; f < folds; f++) {
            int ff = f;
            int[] in = java.util.stream.IntStream.range(0, n).filter(i -> i % folds != ff).map(i -> perm[i]).toArray();
            int[] outRows = java.util.stream.IntStream.range(0, n).filter(i -> i % folds == ff).map(i -> perm[i])
                    .toArray();
            if (outRows.length == 0) {
                continue;
            }
            double[][] p = scorer.probabilities(rows(x, in), labels(y, in), rows(x, outRows));
            for (int k = 0; k < outRows.length; k++) {
                out[outRows[k]] = p[k];
            }
        }
        for (int i = 0; i < n; i++) {
            if (out[i] == null) {
                out[i] = new double[classes];
                Arrays.fill(out[i], 1.0 / classes);
            }
        }
        return out;
    }

    /**
     * @param probabilities {@code [rows][classes]}
     * @param y labels
     * @return the share of rows whose most probable class is the label
     */
    public static double accuracy(double[][] probabilities, int[] y) {
        int right = 0;
        for (int i = 0; i < y.length; i++) {
            int best = 0;
            for (int c = 1; c < probabilities[i].length; c++) {
                if (probabilities[i][c] > probabilities[i][best]) {
                    best = c;
                }
            }
            right += best == y[i] ? 1 : 0;
        }
        return right / (double) y.length;
    }

    private static double[][] rows(double[][] x, int[] idx) {
        double[][] out = new double[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]];
        }
        return out;
    }

    private static int[] labels(int[] y, int[] idx) {
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = y[idx[i]];
        }
        return out;
    }

    private static int[] permutation(int n, SplittableRandom random) {
        int[] p = new int[n];
        Arrays.setAll(p, i -> i);
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
