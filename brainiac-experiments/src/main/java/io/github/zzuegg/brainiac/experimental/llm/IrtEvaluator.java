package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;

/**
 * Benchmark evaluation by item response theory, after Fluid Benchmarking (Hofmann et al.
 * 2025, <a href="https://arxiv.org/abs/2509.11106">arXiv:2509.11106</a>) and tinyBenchmarks
 * (Polo et al. 2024, <a href="https://arxiv.org/abs/2402.14992">arXiv:2402.14992</a>).
 * Accuracy treats every item the same, but items differ in difficulty and in how well they
 * separate strong from weak models; a few noisy or mislabelled items make accuracy curves
 * jumpy. A two-parameter logistic (2PL) model
 * {@code P(model m answers item j) = sigmoid(a_j (theta_m - b_j))} is fitted on the answers
 * of many models (here: checkpoints), giving each item a difficulty {@code b_j} and a
 * discrimination {@code a_j}, and each model an ability {@code theta_m}.
 *
 * <p>A new model is then evaluated <em>adaptively</em> ({@link #adaptiveAbility}): starting
 * from the prior, it is asked the item with the most Fisher information
 * {@code a_j^2 p (1 - p)} at its current ability estimate, the estimate is updated, and so on
 * for a small budget of items. The reported score is the ability, which weighs informative
 * items more and ignores items that do not discriminate, so it is less noisy and ranks
 * models more consistently than accuracy on the same number of items.
 *
 * <p>Fitting is joint maximum a posteriori estimation by gradient ascent with priors
 * {@code theta ~ N(0, 1)}, {@code b ~ N(0, 2^2)} and {@code log a ~ N(0, 1)}; abilities are
 * found by Newton's method.
 */
public final class IrtEvaluator {

    private final double[] discrimination;
    private final double[] difficulty;

    /**
     * @param discrimination the items' {@code a_j}
     * @param difficulty the items' {@code b_j}
     */
    public IrtEvaluator(double[] discrimination, double[] difficulty) {
        if (discrimination.length != difficulty.length) {
            throw new IllegalArgumentException("one discrimination and one difficulty per item");
        }
        this.discrimination = discrimination.clone();
        this.difficulty = difficulty.clone();
    }

    /**
     * Fits a 2PL model to a response matrix.
     *
     * @param responses {@code [models][items]} whether each model answered each item correctly
     * @param iterations gradient-ascent iterations, for example 500
     * @return the fitted evaluator
     */
    public static IrtEvaluator fit(boolean[][] responses, int iterations) {
        int models = responses.length;
        int items = responses[0].length;
        double[] theta = new double[models];
        double[] logA = new double[items];
        double[] b = new double[items];
        for (int j = 0; j < items; j++) {   // start difficulties at the logit of the error rate
            double right = 0;
            for (boolean[] row : responses) {
                right += row[j] ? 1 : 0;
            }
            double p = (right + 0.5) / (models + 1);
            b[j] = -Math.log(p / (1 - p));
        }
        double lr = 0.5;
        double[] gt = new double[models];
        double[] ga = new double[items];
        double[] gb = new double[items];
        for (int it = 0; it < iterations; it++) {
            java.util.Arrays.fill(gt, 0);
            java.util.Arrays.fill(ga, 0);
            java.util.Arrays.fill(gb, 0);
            for (int m = 0; m < models; m++) {
                for (int j = 0; j < items; j++) {
                    double a = Math.exp(logA[j]);
                    double p = sigmoid(a * (theta[m] - b[j]));
                    double r = (responses[m][j] ? 1 : 0) - p;
                    gt[m] += r * a;
                    gb[j] -= r * a;
                    ga[j] += r * a * (theta[m] - b[j]);   // d/d log a = a * d/da
                }
            }
            for (int m = 0; m < models; m++) {
                theta[m] += lr * (gt[m] - theta[m]) / items;
            }
            for (int j = 0; j < items; j++) {
                b[j] += lr * (gb[j] - b[j] / 4) / models;
                logA[j] += lr * (ga[j] - logA[j]) / models;
                logA[j] = Math.max(-3, Math.min(2, logA[j]));
            }
        }
        double[] a = new double[items];
        for (int j = 0; j < items; j++) {
            a[j] = Math.exp(logA[j]);
        }
        return new IrtEvaluator(a, b);
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /** @return the number of items */
    public int items() {
        return difficulty.length;
    }

    /** @return the items' discriminations; a copy */
    public double[] discrimination() {
        return discrimination.clone();
    }

    /** @return the items' difficulties; a copy */
    public double[] difficulty() {
        return difficulty.clone();
    }

    /**
     * @param item an item
     * @param theta an ability
     * @return the item's Fisher information at that ability
     */
    public double information(int item, double theta) {
        double p = sigmoid(discrimination[item] * (theta - difficulty[item]));
        return discrimination[item] * discrimination[item] * p * (1 - p);
    }

    /**
     * The MAP ability from answers to some items, with a standard normal prior.
     *
     * @param items the items asked
     * @param correct whether each was answered correctly
     * @return the ability estimate
     */
    public double ability(int[] items, boolean[] correct) {
        double theta = 0;
        for (int it = 0; it < 50; it++) {
            double g = -theta;
            double h = -1;
            for (int k = 0; k < items.length; k++) {
                int j = items[k];
                double a = discrimination[j];
                double p = sigmoid(a * (theta - difficulty[j]));
                g += a * ((correct[k] ? 1 : 0) - p);
                h -= a * a * p * (1 - p);
            }
            double step = g / h;
            theta -= Math.max(-1, Math.min(1, step));
            if (Math.abs(step) < 1e-8) {
                break;
            }
        }
        return theta;
    }

    /**
     * Evaluates a model adaptively on {@code budget} items, each chosen for the most
     * information at the current estimate.
     *
     * @param answers whether the model answers an item correctly
     * @param budget the items asked
     * @return the ability estimate
     */
    public double adaptiveAbility(IntPredicate answers, int budget) {
        boolean[] asked = new boolean[items()];
        List<Integer> chosen = new ArrayList<>();
        List<Boolean> right = new ArrayList<>();
        double theta = 0;
        for (int k = 0; k < Math.min(budget, items()); k++) {
            int best = -1;
            for (int j = 0; j < items(); j++) {
                if (!asked[j] && (best < 0 || information(j, theta) > information(best, theta))) {
                    best = j;
                }
            }
            asked[best] = true;
            chosen.add(best);
            right.add(answers.test(best));
            int[] ids = chosen.stream().mapToInt(Integer::intValue).toArray();
            boolean[] ok = new boolean[right.size()];
            for (int i = 0; i < ok.length; i++) {
                ok[i] = right.get(i);
            }
            theta = ability(ids, ok);
        }
        return theta;
    }
}
