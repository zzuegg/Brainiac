package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Robust linear regression, for data where a few rows are wrong.
 *
 * <ul>
 *   <li>{@link Method#HUBER} (Huber, Annals 1964): IRLS with weights {@code min(1, c / |r / s|)},
 *       {@code c = 1.345}, the scale {@code s} from the MAD of the residuals; monotone, so it
 *       resists vertical outliers but not leverage points.</li>
 *   <li>{@link Method#TUKEY}: the redescending bisquare, {@code w = (1 - (r / (c s))^2)^2} for
 *       {@code |r| < c s}, {@code c = 4.685}, started from the Huber fit.</li>
 *   <li>{@link Method#LTS}: FAST-LTS (Rousseeuw and Van Driessen, DMKD 2006): least squares on
 *       the {@code h = (n + p + 1) / 2} rows with the smallest residuals, found by concentration
 *       steps from random elemental starts; 50% breakdown.</li>
 *   <li>{@link Method#THEIL_SEN} (Sen, JASA 1968): the median of pairwise slopes for one feature;
 *       for several, the coordinate-wise median of exact fits on random elemental subsets.</li>
 *   <li>{@link Method#MM} (Yohai 1987): LTS for a high-breakdown start and scale, then Tukey
 *       IRLS with that scale fixed for efficiency.</li>
 * </ul>
 */
public final class RobustRegression {

    /** The estimator. */
    public enum Method {
        /** Huber M-estimator. */
        HUBER,
        /** Tukey bisquare M-estimator. */
        TUKEY,
        /** Least trimmed squares. */
        LTS,
        /** Theil-Sen. */
        THEIL_SEN,
        /** MM-estimator from an LTS start. */
        MM
    }

    /**
     * A fit.
     *
     * @param intercept the intercept
     * @param beta the slopes
     * @param scale the robust residual scale
     */
    public record Fit(double intercept, double[] beta, double scale) {

        /**
         * Residuals {@code y - b0 - x b}.
         *
         * @param x rows
         * @param y targets
         * @return the residuals
         */
        public double[] residuals(double[][] x, double[] y) {
            double[] r = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                double f = intercept;
                for (int j = 0; j < beta.length; j++) {
                    f += x[i][j] * beta[j];
                }
                r[i] = y[i] - f;
            }
            return r;
        }
    }

    private RobustRegression() {
    }

    /**
     * Fits a robust regression.
     *
     * @param method the estimator
     * @param x the design {@code [n][p]}
     * @param y the response
     * @param seed the seed for random starts and subsets
     * @return the fit
     */
    public static Fit fit(Method method, double[][] x, double[] y, long seed) {
        return switch (method) {
            case HUBER -> irls(x, y, ols(x, y), true, 1.345, Double.NaN);
            case TUKEY -> {
                Fit h = irls(x, y, ols(x, y), true, 1.345, Double.NaN);
                yield irls(x, y, h, false, 4.685, Double.NaN);
            }
            case LTS -> lts(x, y, seed);
            case THEIL_SEN -> theilSen(x, y, seed);
            case MM -> {
                Fit start = lts(x, y, seed);
                yield irls(x, y, start, false, 4.685, start.scale());
            }
        };
    }

    /**
     * Ordinary least squares as a fit.
     *
     * @param x the design
     * @param y the response
     * @return the OLS fit
     */
    public static Fit ols(double[][] x, double[] y) {
        LeastSquares.Fit f = LeastSquares.fit(x, y, true, 1e-9);
        double[] b = Arrays.copyOfRange(f.beta(), 1, f.beta().length);
        return new Fit(f.beta()[0], b, mad(f.residuals()));
    }

    /**
     * The normalised median absolute deviation from zero, {@code median|r| / 0.6745}.
     *
     * @param r residuals
     * @return the scale
     */
    public static double mad(double[] r) {
        double[] a = new double[r.length];
        for (int i = 0; i < r.length; i++) {
            a[i] = Math.abs(r[i]);
        }
        return Math.max(1e-12, median(a) / 0.6745);
    }

    /**
     * The median.
     *
     * @param v values (not modified)
     * @return the median
     */
    public static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        int n = s.length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    private static Fit irls(double[][] x, double[] y, Fit start, boolean huber, double c, double fixedScale) {
        int n = y.length;
        int p = x[0].length;
        Fit cur = start;
        for (int it = 0; it < 100; it++) {
            double[] r = cur.residuals(x, y);
            double s = Double.isNaN(fixedScale) ? mad(r) : fixedScale;
            double[] w = new double[n];
            for (int i = 0; i < n; i++) {
                double u = Math.abs(r[i]) / (c * s);
                w[i] = huber ? Math.min(1, 1 / Math.max(u, 1e-12)) : u < 1 ? (1 - u * u) * (1 - u * u) : 0;
            }
            Fit next = weighted(x, y, w);
            double change = Math.abs(next.intercept() - cur.intercept());
            for (int j = 0; j < p; j++) {
                change = Math.max(change, Math.abs(next.beta()[j] - cur.beta()[j]));
            }
            cur = new Fit(next.intercept(), next.beta(), s);
            if (change < 1e-8) {
                break;
            }
        }
        return cur;
    }

    /**
     * Weighted least squares.
     *
     * @param x the design
     * @param y the response
     * @param w row weights
     * @return the fit (scale NaN)
     */
    public static Fit weighted(double[][] x, double[] y, double[] w) {
        int n = y.length;
        int p = x[0].length;
        double[][] g = new double[p + 1][p + 1];
        double[] b = new double[p + 1];
        double[] z = new double[p + 1];
        for (int i = 0; i < n; i++) {
            if (w[i] == 0) {
                continue;
            }
            z[0] = 1;
            System.arraycopy(x[i], 0, z, 1, p);
            for (int a = 0; a <= p; a++) {
                b[a] += w[i] * z[a] * y[i];
                for (int d = 0; d <= p; d++) {
                    g[a][d] += w[i] * z[a] * z[d];
                }
            }
        }
        for (int a = 1; a <= p; a++) {
            g[a][a] += 1e-9;
        }
        double[] sol = io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators.solve(g, b);
        return new Fit(sol[0], Arrays.copyOfRange(sol, 1, p + 1), Double.NaN);
    }

    private static Fit lts(double[][] x, double[] y, long seed) {
        int n = y.length;
        int p = x[0].length;
        int h = (n + p + 2) / 2;
        SplittableRandom r = new SplittableRandom(seed);
        Fit best = null;
        double bestObj = Double.POSITIVE_INFINITY;
        int starts = 200;
        Fit[] candidates = new Fit[starts];
        double[] objs = new double[starts];
        for (int s = 0; s < starts; s++) {
            double[] w = new double[n];
            int[] idx = sample(n, p + 1, r);
            for (int i : idx) {
                w[i] = 1;
            }
            Fit f = weighted(x, y, w);
            for (int c = 0; c < 2; c++) {
                f = cStep(x, y, f, h);
            }
            candidates[s] = f;
            objs[s] = trimmed(f, x, y, h);
        }
        Integer[] order = new Integer[starts];
        for (int s = 0; s < starts; s++) {
            order[s] = s;
        }
        Arrays.sort(order, (a, b) -> Double.compare(objs[a], objs[b]));
        for (int k = 0; k < 10; k++) {
            Fit f = candidates[order[k]];
            double prev = Double.POSITIVE_INFINITY;
            for (int c = 0; c < 50; c++) {
                f = cStep(x, y, f, h);
                double o = trimmed(f, x, y, h);
                if (o >= prev - 1e-12) {
                    break;
                }
                prev = o;
            }
            double o = trimmed(f, x, y, h);
            if (o < bestObj) {
                bestObj = o;
                best = f;
            }
        }
        // consistency-corrected scale of the h smallest squared residuals
        double s0 = Math.sqrt(bestObj / h);
        double[] res = best.residuals(x, y);
        double sc = s0 * 1.0 / Math.sqrt(Math.max(1e-3, consistency((double) h / n)));
        // reweighting step at 2.5 scales
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = Math.abs(res[i]) <= 2.5 * sc ? 1 : 0;
        }
        Fit rw = weighted(x, y, w);
        return new Fit(rw.intercept(), rw.beta(), sc);
    }

    /** E[Z^2 | |Z| <= q] for the h/n quantile q, the LTS scale consistency factor. */
    private static double consistency(double share) {
        double q = io.github.zzuegg.brainiac.experimental.ml.NormalTails.quantile(0.5 + share / 2);
        double phi = Math.exp(-q * q / 2) / Math.sqrt(2 * Math.PI);
        return 1 - 2 * q * phi / share;
    }

    private static Fit cStep(double[][] x, double[] y, Fit f, int h) {
        double[] r = f.residuals(x, y);
        Integer[] o = new Integer[r.length];
        for (int i = 0; i < r.length; i++) {
            o[i] = i;
        }
        Arrays.sort(o, (a, b) -> Double.compare(Math.abs(r[a]), Math.abs(r[b])));
        double[] w = new double[r.length];
        for (int k = 0; k < h; k++) {
            w[o[k]] = 1;
        }
        return weighted(x, y, w);
    }

    private static double trimmed(Fit f, double[][] x, double[] y, int h) {
        double[] r = f.residuals(x, y);
        for (int i = 0; i < r.length; i++) {
            r[i] *= r[i];
        }
        Arrays.sort(r);
        double s = 0;
        for (int k = 0; k < h; k++) {
            s += r[k];
        }
        return s;
    }

    private static Fit theilSen(double[][] x, double[] y, long seed) {
        int n = y.length;
        int p = x[0].length;
        if (p == 1) {
            double[] slopes = new double[n * (n - 1) / 2];
            int k = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double dx = x[j][0] - x[i][0];
                    slopes[k++] = dx == 0 ? 0 : (y[j] - y[i]) / dx;
                }
            }
            double b = median(Arrays.copyOf(slopes, k));
            double[] icpt = new double[n];
            for (int i = 0; i < n; i++) {
                icpt[i] = y[i] - b * x[i][0];
            }
            Fit f = new Fit(median(icpt), new double[] {b}, 0);
            return new Fit(f.intercept(), f.beta(), mad(f.residuals(x, y)));
        }
        SplittableRandom r = new SplittableRandom(seed);
        int draws = 1000;
        double[][] coef = new double[p + 1][draws];
        for (int d = 0; d < draws; d++) {
            double[] w = new double[n];
            for (int i : sample(n, p + 1, r)) {
                w[i] = 1;
            }
            Fit f = weighted(x, y, w);
            coef[0][d] = f.intercept();
            for (int j = 0; j < p; j++) {
                coef[j + 1][d] = f.beta()[j];
            }
        }
        double[] b = new double[p];
        for (int j = 0; j < p; j++) {
            b[j] = median(coef[j + 1]);
        }
        Fit f = new Fit(median(coef[0]), b, 0);
        return new Fit(f.intercept(), b, mad(f.residuals(x, y)));
    }

    private static int[] sample(int n, int k, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = 0; i < k; i++) {
            int j = i + r.nextInt(n - i);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return Arrays.copyOf(a, k);
    }
}
