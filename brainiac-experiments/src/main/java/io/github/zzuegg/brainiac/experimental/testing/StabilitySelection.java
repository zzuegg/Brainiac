package io.github.zzuegg.brainiac.experimental.testing;

import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;

import java.util.SplittableRandom;

/**
 * Stability selection (Meinshausen and Buhlmann, JRSS-B 2010,
 * <a href="https://arxiv.org/abs/0809.2932">arXiv:0809.2932</a>): refit the lasso on many random
 * half-samples over a short lambda grid and keep the features whose selection frequency (the
 * maximum over the grid) reaches {@code piThreshold}. With {@code q} features selected per fit
 * on average, the expected number of false selections is at most
 * {@code q^2 / ((2 piThreshold - 1) p)}.
 */
public final class StabilitySelection {

    /**
     * A result.
     *
     * @param frequency the selection frequency of every feature
     * @param selected the stable set
     */
    public record Result(double[] frequency, boolean[] selected) {
    }

    private StabilitySelection() {
    }

    /**
     * Runs stability selection with the lasso.
     *
     * @param x the design {@code [n][p]}
     * @param y the response
     * @param subsamples the number of half-samples
     * @param lambdas grid points, from {@code lambda_max} down to {@code ratio lambda_max}
     * @param ratio the smallest lambda share
     * @param piThreshold the frequency threshold, in {@code (0.5, 1]}
     * @param seed the seed
     * @return the selection
     */
    public static Result select(double[][] x, double[] y, int subsamples, int lambdas, double ratio,
            double piThreshold, long seed) {
        int n = y.length;
        int p = x[0].length;
        PenalizedRegression lasso = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO);
        double[] grid = PenalizedRegression.grid(lasso.lambdaMax(x, y), lambdas, ratio);
        int[][] counts = new int[lambdas][p];
        SplittableRandom r = new SplittableRandom(seed);
        int half = n / 2;
        for (int b = 0; b < subsamples; b++) {
            int[] idx = new int[n];
            for (int i = 0; i < n; i++) {
                idx[i] = i;
            }
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = idx[i];
                idx[i] = idx[j];
                idx[j] = t;
            }
            double[][] xs = new double[half][];
            double[] ys = new double[half];
            for (int i = 0; i < half; i++) {
                xs[i] = x[idx[i]];
                ys[i] = y[idx[i]];
            }
            PenalizedRegression.Path path = lasso.path(xs, ys, grid);
            for (int k = 0; k < lambdas; k++) {
                double[] beta = path.fits().get(k).beta();
                for (int j = 0; j < p; j++) {
                    if (beta[j] != 0) {
                        counts[k][j]++;
                    }
                }
            }
        }
        double[] freq = new double[p];
        boolean[] sel = new boolean[p];
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < lambdas; k++) {
                freq[j] = Math.max(freq[j], counts[k][j] / (double) subsamples);
            }
            sel[j] = freq[j] >= piThreshold;
        }
        return new Result(freq, sel);
    }
}
