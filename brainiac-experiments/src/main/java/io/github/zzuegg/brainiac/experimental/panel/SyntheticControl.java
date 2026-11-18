package io.github.zzuegg.brainiac.experimental.panel;

import java.util.Arrays;

/**
 * Synthetic control (Abadie, Diamond and Hainmueller, JASA 2010,
 * <a href="https://doi.org/10.1198/jasa.2009.ap08746">doi:10.1198/jasa.2009.ap08746</a>): the
 * treated unit's counterfactual is a convex combination of donor units, with weights that make
 * it track the treated unit's pre-treatment outcomes.
 *
 * <p>The weights solve {@code min ||y_pre - Y0_pre w||^2 + ridge ||w||^2} over the simplex by
 * accelerated projected gradient (FISTA with the exact Euclidean projection onto the simplex of
 * Duchi et al. 2008), optionally with a free intercept (used by {@link SyntheticDid}). A block of
 * treated units is handled through their average. Inference is placebo-in-space: the estimator is
 * re-run with each donor as a fake treated unit, and the p-value is the rank of the treated
 * unit's post/pre RMSPE ratio among all.
 */
public final class SyntheticControl {

    private SyntheticControl() {
    }

    /**
     * An estimate.
     *
     * @param effect the average post-period gap between the treated outcome and its synthetic
     *     counterfactual
     * @param weights donor weights
     * @param preRmse the pre-period fit
     */
    public record Estimate(double effect, double[] weights, double preRmse) {
    }

    /**
     * Euclidean projection onto the probability simplex.
     *
     * @param v a vector
     * @return the closest point with non-negative entries summing to 1
     */
    public static double[] projectSimplex(double[] v) {
        int n = v.length;
        double[] u = v.clone();
        Arrays.sort(u);
        double css = 0;
        double theta = 0;
        for (int k = n - 1; k >= 0; k--) {
            css += u[k];
            double t = (css - 1) / (n - k);
            if (k == 0 || u[k - 1] <= t) {
                theta = t;
                break;
            }
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = Math.max(0, v[i] - theta);
        }
        return w;
    }

    /**
     * Simplex-constrained least squares {@code min_{c, w} ||b - c - A w||^2 + ridge ||w||^2}.
     *
     * @param a the design {@code [rows][k]} (columns are the candidates)
     * @param b the target
     * @param ridge the ridge on the weights
     * @param intercept whether {@code c} is free (else 0)
     * @param iterations FISTA iterations
     * @return the weights (the intercept is not returned)
     */
    public static double[] simplexLeastSquares(double[][] a, double[] b, double ridge, boolean intercept,
            int iterations) {
        int m = a.length;
        int k = a[0].length;
        double[][] x = new double[m][k];
        double[] y = b.clone();
        if (intercept) {
            double[] mean = new double[k];
            double my = 0;
            for (int r = 0; r < m; r++) {
                my += b[r] / m;
                for (int j = 0; j < k; j++) {
                    mean[j] += a[r][j] / m;
                }
            }
            for (int r = 0; r < m; r++) {
                y[r] -= my;
                for (int j = 0; j < k; j++) {
                    x[r][j] = a[r][j] - mean[j];
                }
            }
        } else {
            for (int r = 0; r < m; r++) {
                x[r] = a[r].clone();
            }
        }
        // Lipschitz constant of the gradient by power iteration on X^T X + ridge
        double[] v = new double[k];
        Arrays.fill(v, 1 / Math.sqrt(k));
        double lip = 1;
        for (int it = 0; it < 50; it++) {
            double[] xv = new double[m];
            for (int r = 0; r < m; r++) {
                for (int j = 0; j < k; j++) {
                    xv[r] += x[r][j] * v[j];
                }
            }
            double[] g = new double[k];
            for (int r = 0; r < m; r++) {
                for (int j = 0; j < k; j++) {
                    g[j] += x[r][j] * xv[r];
                }
            }
            double norm = 0;
            for (double q : g) {
                norm += q * q;
            }
            norm = Math.sqrt(norm);
            if (norm == 0) {
                break;
            }
            lip = norm;
            for (int j = 0; j < k; j++) {
                v[j] = g[j] / norm;
            }
        }
        lip = 2 * (lip + ridge) * 1.01;
        double[] w = new double[k];
        Arrays.fill(w, 1.0 / k);
        double[] z = w.clone();
        double tk = 1;
        for (int it = 0; it < iterations; it++) {
            double[] res = new double[m];
            for (int r = 0; r < m; r++) {
                double f = 0;
                for (int j = 0; j < k; j++) {
                    f += x[r][j] * z[j];
                }
                res[r] = f - y[r];
            }
            double[] grad = new double[k];
            for (int j = 0; j < k; j++) {
                grad[j] = 2 * ridge * z[j];
            }
            for (int r = 0; r < m; r++) {
                for (int j = 0; j < k; j++) {
                    grad[j] += 2 * x[r][j] * res[r];
                }
            }
            double[] step = new double[k];
            for (int j = 0; j < k; j++) {
                step[j] = z[j] - grad[j] / lip;
            }
            double[] next = projectSimplex(step);
            double tn = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
            for (int j = 0; j < k; j++) {
                z[j] = next[j] + (tk - 1) / tn * (next[j] - w[j]);
            }
            w = next;
            tk = tn;
        }
        return w;
    }

    /**
     * Fits synthetic control.
     *
     * @param y outcomes {@code [units][periods]}
     * @param treatedUnits the treated units (averaged into one series)
     * @param start the first treated period
     * @return the estimate
     */
    public static Estimate fit(double[][] y, int[] treatedUnits, int start) {
        int periods = y[0].length;
        double[] target = new double[periods];
        for (int i : treatedUnits) {
            for (int t = 0; t < periods; t++) {
                target[t] += y[i][t] / treatedUnits.length;
            }
        }
        int[] donors = complement(y.length, treatedUnits);
        double[][] a = new double[start][donors.length];
        for (int t = 0; t < start; t++) {
            for (int j = 0; j < donors.length; j++) {
                a[t][j] = y[donors[j]][t];
            }
        }
        double[] w = simplexLeastSquares(a, Arrays.copyOf(target, start), 1e-6, false, 500);
        double pre = 0;
        double post = 0;
        for (int t = 0; t < periods; t++) {
            double synth = 0;
            for (int j = 0; j < donors.length; j++) {
                synth += w[j] * y[donors[j]][t];
            }
            double gap = target[t] - synth;
            if (t < start) {
                pre += gap * gap / start;
            } else {
                post += gap / (periods - start);
            }
        }
        return new Estimate(post, w, Math.sqrt(pre));
    }

    /**
     * The units not in a set.
     *
     * @param units all units {@code 0..units-1}
     * @param excluded the excluded ones
     * @return the rest, ascending
     */
    public static int[] complement(int units, int[] excluded) {
        boolean[] ex = new boolean[units];
        for (int i : excluded) {
            ex[i] = true;
        }
        int[] out = new int[units - excluded.length];
        int c = 0;
        for (int i = 0; i < units; i++) {
            if (!ex[i]) {
                out[c++] = i;
            }
        }
        return out;
    }

    /**
     * The placebo-in-space p-value: the rank of the treated unit's post/pre RMSPE ratio among
     * the ratios of every donor used as a fake treated unit (with the true treated units
     * dropped from the donor pool).
     *
     * @param y outcomes
     * @param treatedUnits the treated units
     * @param start the first treated period
     * @return the p-value in {@code (0, 1]}
     */
    public static double placeboPValue(double[][] y, int[] treatedUnits, int start) {
        double real = ratio(y, treatedUnits, start);
        int[] donors = complement(y.length, treatedUnits);
        double[][] pool = new double[donors.length][];
        for (int j = 0; j < donors.length; j++) {
            pool[j] = y[donors[j]];
        }
        int above = 1;
        for (int j = 0; j < donors.length; j++) {
            if (ratio(pool, new int[] {j}, start) >= real) {
                above++;
            }
        }
        return above / (double) (donors.length + 1);
    }

    private static double ratio(double[][] y, int[] treated, int start) {
        Estimate e = fit(y, treated, start);
        int periods = y[0].length;
        int[] donors = complement(y.length, treated);
        double post = 0;
        for (int t = start; t < periods; t++) {
            double target = 0;
            for (int i : treated) {
                target += y[i][t] / treated.length;
            }
            double synth = 0;
            for (int j = 0; j < donors.length; j++) {
                synth += e.weights()[j] * y[donors[j]][t];
            }
            post += (target - synth) * (target - synth) / (periods - start);
        }
        return Math.sqrt(post) / Math.max(e.preRmse(), 1e-9);
    }
}
