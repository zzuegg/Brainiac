package io.github.zzuegg.brainiac.experimental.tsc;

import io.github.zzuegg.brainiac.experimental.ml.RandomForest;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Ways to combine series classifiers, and the measure of how much they leave on the table.
 * <ul>
 *   <li>{@link #vote}: majority voting of the members' predictions;</li>
 *   <li>{@link #cawpe}: the Cross-validation Accuracy Weighted Probabilistic Ensemble (Large,
 *       Lines and Bagnall, DMKD 2019): member probabilities weighted by their cross-validated
 *       accuracy raised to {@code alpha} (4 in the paper), which sharpens the gap between good
 *       and mediocre members;</li>
 *   <li>{@link #stack}: symmetric stacking, a meta-learner on every member's out-of-fold
 *       scores;</li>
 *   <li>{@link #asymmetricStack}: one member's features plus the other members' out-of-fold
 *       scores (QUANT's quantiles with Hydra's logits in Maniar, Dec 2025,
 *       <a href="https://arxiv.org/abs/2512.06666">arXiv:2512.06666</a>);</li>
 *   <li>{@link #concatenate}: a meta-learner on the members' raw feature blocks side by side;</li>
 *   <li>{@link #oracle} and {@link #gapClosed}: the accuracy of an oracle that picks the right
 *       member for every example whenever one exists, and the share of the gap between the best
 *       single member and that oracle an ensemble closes (the "meta-learning gap").</li>
 * </ul>
 * Out-of-fold scores come from {@link #outOfFold}, which refits a {@link Member} on k-1 folds.
 */
public final class SeriesEnsembles {

    private SeriesEnsembles() {
    }

    /** A classifier that fits on training series and scores test series. */
    @FunctionalInterface
    public interface Member {
        /**
         * Fits and scores.
         *
         * @param trainX training series
         * @param trainY their classes
         * @param classes the number of classes
         * @param testX series to score
         * @return {@code [testX.length][classes]} scores, larger for more likely classes
         */
        double[][] fitScores(float[][] trainX, int[] trainY, int classes, float[][] testX);
    }

    /** The meta-learner of the stacking strategies. */
    public enum Meta {
        /** Extremely randomised trees. */
        EXTRA_TREES,
        /** A ridge classifier with leave-one-out penalty selection. */
        RIDGE
    }

    /**
     * Out-of-fold scores: every training series scored by a member fitted without its fold.
     *
     * @param member the member
     * @param x training series
     * @param y their classes
     * @param classes the number of classes
     * @param folds the number of folds, such as 5
     * @param seed the fold assignment seed
     * @return {@code [n][classes]}
     */
    public static double[][] outOfFold(Member member, float[][] x, int[] y, int classes, int folds, long seed) {
        int n = x.length;
        int[] fold = new int[n];
        int[] perm = permutation(n, new SplittableRandom(seed));
        for (int i = 0; i < n; i++) {
            fold[perm[i]] = i % folds;
        }
        double[][] out = new double[n][];
        for (int f = 0; f < folds; f++) {
            int nTest = 0;
            for (int i = 0; i < n; i++) {
                nTest += fold[i] == f ? 1 : 0;
            }
            float[][] tx = new float[n - nTest][];
            int[] ty = new int[n - nTest];
            float[][] vx = new float[nTest][];
            int[] vi = new int[nTest];
            int a = 0;
            int b = 0;
            for (int i = 0; i < n; i++) {
                if (fold[i] == f) {
                    vx[b] = x[i];
                    vi[b++] = i;
                } else {
                    tx[a] = x[i];
                    ty[a++] = y[i];
                }
            }
            double[][] s = member.fitScores(tx, ty, classes, vx);
            for (int k = 0; k < nTest; k++) {
                out[vi[k]] = s[k];
            }
        }
        return out;
    }

    /**
     * Row-wise softmax of scores with a temperature (probabilities stay probabilities only if
     * the temperature is chosen for them; for CAWPE, use it on decision values).
     *
     * @param scores {@code [n][classes]}
     * @param temperature the temperature, positive
     * @return probabilities
     */
    public static double[][] softmax(double[][] scores, double temperature) {
        double[][] out = new double[scores.length][];
        for (int i = 0; i < scores.length; i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (double v : scores[i]) {
                max = Math.max(max, v);
            }
            double sum = 0;
            out[i] = new double[scores[i].length];
            for (int c = 0; c < out[i].length; c++) {
                out[i][c] = Math.exp((scores[i][c] - max) / temperature);
                sum += out[i][c];
            }
            for (int c = 0; c < out[i].length; c++) {
                out[i][c] /= sum;
            }
        }
        return out;
    }

    /**
     * The accuracy of the argmax of scores.
     *
     * @param scores {@code [n][classes]}
     * @param truth the classes
     * @return the accuracy
     */
    public static double accuracy(double[][] scores, int[] truth) {
        int[] p = WideRidgeClassifier.argmax(scores);
        int right = 0;
        for (int i = 0; i < truth.length; i++) {
            right += p[i] == truth[i] ? 1 : 0;
        }
        return right / (double) truth.length;
    }

    /**
     * Majority vote (ties to the member listed first among the tied classes' voters).
     *
     * @param testScores each member's test scores
     * @return vote shares {@code [n][classes]}
     */
    public static double[][] vote(List<double[][]> testScores) {
        int n = testScores.get(0).length;
        int classes = testScores.get(0)[0].length;
        double[][] out = new double[n][classes];
        for (int m = 0; m < testScores.size(); m++) {
            int[] p = WideRidgeClassifier.argmax(testScores.get(m));
            for (int i = 0; i < n; i++) {
                out[i][p[i]] += 1 + 1e-6 * (testScores.size() - m);
            }
        }
        return out;
    }

    /**
     * CAWPE: probabilities weighted by cross-validated accuracy to the power {@code alpha}.
     *
     * @param oofProbabilities each member's out-of-fold training probabilities
     * @param testProbabilities each member's test probabilities
     * @param labels the training classes
     * @param alpha the exponent, 4 in the paper
     * @return combined scores {@code [n][classes]}
     */
    public static double[][] cawpe(List<double[][]> oofProbabilities, List<double[][]> testProbabilities,
            int[] labels, double alpha) {
        int n = testProbabilities.get(0).length;
        int classes = testProbabilities.get(0)[0].length;
        double[][] out = new double[n][classes];
        for (int m = 0; m < testProbabilities.size(); m++) {
            double w = Math.pow(accuracy(oofProbabilities.get(m), labels), alpha);
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < classes; c++) {
                    out[i][c] += w * testProbabilities.get(m)[i][c];
                }
            }
        }
        return out;
    }

    /**
     * Symmetric stacking: a meta-learner on every member's out-of-fold scores.
     *
     * @param oofScores each member's out-of-fold training scores
     * @param testScores each member's test scores
     * @param labels the training classes
     * @param classes the number of classes
     * @param meta the meta-learner
     * @param seed the meta-learner seed
     * @return the meta-learner's test scores
     */
    public static double[][] stack(List<double[][]> oofScores, List<double[][]> testScores, int[] labels,
            int classes, Meta meta, long seed) {
        double[][] a = oofScores.get(0);
        double[][] b = testScores.get(0);
        for (int m = 1; m < oofScores.size(); m++) {
            a = Hydra.concatenate(a, oofScores.get(m));
            b = Hydra.concatenate(b, testScores.get(m));
        }
        return metaScores(a, labels, classes, b, meta, seed);
    }

    /**
     * Asymmetric stacking: one member's features with the other members' out-of-fold scores.
     *
     * @param trainFeatures the base member's training features
     * @param testFeatures its test features
     * @param oofScores the other members' out-of-fold training scores
     * @param testScores their test scores
     * @param labels the training classes
     * @param classes the number of classes
     * @param meta the meta-learner
     * @param seed the seed
     * @return the meta-learner's test scores
     */
    public static double[][] asymmetricStack(double[][] trainFeatures, double[][] testFeatures,
            List<double[][]> oofScores, List<double[][]> testScores, int[] labels, int classes, Meta meta,
            long seed) {
        double[][] a = trainFeatures;
        double[][] b = testFeatures;
        for (int m = 0; m < oofScores.size(); m++) {
            a = Hydra.concatenate(a, oofScores.get(m));
            b = Hydra.concatenate(b, testScores.get(m));
        }
        return metaScores(a, labels, classes, b, meta, seed);
    }

    /**
     * Feature concatenation: a learner on all members' feature blocks side by side.
     *
     * @param trainBlocks the training feature blocks
     * @param testBlocks the test feature blocks
     * @param labels the training classes
     * @param classes the number of classes
     * @param meta the learner
     * @param seed the seed
     * @return test scores
     */
    public static double[][] concatenate(List<double[][]> trainBlocks, List<double[][]> testBlocks, int[] labels,
            int classes, Meta meta, long seed) {
        double[][] a = trainBlocks.get(0);
        double[][] b = testBlocks.get(0);
        for (int m = 1; m < trainBlocks.size(); m++) {
            a = Hydra.concatenate(a, trainBlocks.get(m));
            b = Hydra.concatenate(b, testBlocks.get(m));
        }
        return metaScores(a, labels, classes, b, meta, seed);
    }

    private static double[][] metaScores(double[][] train, int[] labels, int classes, double[][] test, Meta meta,
            long seed) {
        if (meta == Meta.RIDGE) {
            return WideRidgeClassifier.fit(train, labels, classes, WideRidgeClassifier.defaultAlphas()).scores(test);
        }
        int per = Math.max(1, (int) Math.round(Math.max(Math.sqrt(train[0].length), 0.1 * train[0].length)));
        RandomForest f = RandomForest.fit(train, labels, classes,
                new RandomForest.Settings(RandomForest.Mode.EXTRA_TREES, 200, 40, 1, per, seed));
        double[][] out = new double[test.length][];
        for (int i = 0; i < test.length; i++) {
            out[i] = f.predictProbability(test[i]);
        }
        return out;
    }

    /**
     * The oracle accuracy: the share of examples that at least one member gets right.
     *
     * @param testScores each member's test scores
     * @param truth the classes
     * @return the oracle accuracy
     */
    public static double oracle(List<double[][]> testScores, int[] truth) {
        boolean[] right = new boolean[truth.length];
        for (double[][] s : testScores) {
            int[] p = WideRidgeClassifier.argmax(s);
            for (int i = 0; i < truth.length; i++) {
                right[i] |= p[i] == truth[i];
            }
        }
        int k = 0;
        for (boolean b : right) {
            k += b ? 1 : 0;
        }
        return k / (double) truth.length;
    }

    /**
     * The share of the gap between the best member and the oracle that an ensemble closes.
     *
     * @param ensemble the ensemble's accuracy
     * @param best the best single member's accuracy
     * @param oracle the oracle accuracy
     * @return {@code (ensemble - best) / (oracle - best)}, 0 when there is no gap
     */
    public static double gapClosed(double ensemble, double best, double oracle) {
        return oracle - best < 1e-12 ? 0 : (ensemble - best) / (oracle - best);
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
