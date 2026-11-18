package io.github.zzuegg.brainiac.experimental.panel;

import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;

import java.util.Arrays;

/**
 * Regression discontinuity designs: when treatment switches at a cutoff {@code c} of a running
 * variable, the jump of the outcome's conditional mean at {@code c} identifies a local effect.
 *
 * <ul>
 *   <li>{@link #sharp}: local polynomial regression with a triangular kernel on each side; every
 *       estimate is a linear smoother {@code sum l_i y_i}, so its variance is
 *       {@code sum l_i^2 sigma_i^2} with nearest-neighbour residual variances (3 neighbours, as
 *       in rdrobust).</li>
 *   <li>Robust bias correction (Calonico, Cattaneo and Titiunik, Econometrica 2014,
 *       <a href="https://doi.org/10.3982/ECTA11757">doi:10.3982/ECTA11757</a>): the leading
 *       bias of the local linear fit, {@code (m''/2) e1' Gamma^-1 sum w_i X_i (x_i - c)^2}, is
 *       estimated with a local quadratic at a pilot bandwidth {@code b} and subtracted; since the
 *       correction is linear in {@code y} too, the robust variance counts its own variability,
 *       which is what restores coverage.</li>
 *   <li>Bandwidths: {@link #ikBandwidth}, the MSE-optimal rule of Imbens and Kalyanaraman (REStud
 *       2012, <a href="https://doi.org/10.1093/restud/rdr043">doi:10.1093/restud/rdr043</a>) with
 *       its regularisation, and {@link #cctBandwidths}, a CCT-style plug-in for the pair
 *       {@code (h, b)}.</li>
 *   <li>{@link #fuzzy}: the local Wald ratio of the outcome jump to the take-up jump, with a
 *       delta-method standard error.</li>
 *   <li>{@link #densityTest}: McCrary's test (J. Econometrics 2008,
 *       <a href="https://doi.org/10.1016/j.jeconom.2007.05.005">doi:10.1016/j.jeconom.2007.05.005</a>)
 *       for a jump of the running variable's log density at the cutoff (manipulation).</li>
 *   <li>Baselines: {@link #globalPolynomial} of order 1 to 4 on each side and
 *       {@link #windowMeans}.</li>
 * </ul>
 */
public final class RegressionDiscontinuity {

    private RegressionDiscontinuity() {
    }

    /**
     * An estimate with its standard error and 95% interval.
     *
     * @param effect the estimate
     * @param se the standard error
     * @param low the lower 95% limit
     * @param high the upper 95% limit
     */
    public record Estimate(double effect, double se, double low, double high) {

        static Estimate of(double effect, double se) {
            return new Estimate(effect, se, effect - 1.959964 * se, effect + 1.959964 * se);
        }

        /**
         * @param truth the true value
         * @return whether the interval covers it
         */
        public boolean covers(double truth) {
            return low <= truth && truth <= high;
        }
    }

    /**
     * The linear-smoother weights of a local polynomial's coefficient at the cutoff, one side.
     *
     * @param x the running variable
     * @param c the cutoff
     * @param h the bandwidth
     * @param order the polynomial order
     * @param above which side
     * @param coefficient which coefficient (0 the intercept, 2 the quadratic term)
     * @return per-unit weights (0 outside the side or window)
     */
    public static double[] smoother(double[] x, double c, double h, int order, boolean above, int coefficient) {
        int n = x.length;
        int p = order + 1;
        double[][] gram = new double[p][p];
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double u = (x[i] - c) / h;
            boolean side = above ? x[i] >= c : x[i] < c;
            if (!side || Math.abs(u) >= 1) {
                continue;
            }
            w[i] = 1 - Math.abs(u);
            double[] r = powers(x[i] - c, p);
            for (int a = 0; a < p; a++) {
                for (int b = 0; b < p; b++) {
                    gram[a][b] += w[i] * r[a] * r[b];
                }
            }
        }
        double[] e = new double[p];
        e[coefficient] = 1;
        double[] g = MlMatrices.choleskySolve(MlMatrices.cholesky(gram), e);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            if (w[i] > 0) {
                out[i] = w[i] * MlMatrices.dot(g, powers(x[i] - c, p));
            }
        }
        return out;
    }

    private static double[] powers(double v, int p) {
        double[] r = new double[p];
        double q = 1;
        for (int a = 0; a < p; a++) {
            r[a] = q;
            q *= v;
        }
        return r;
    }

    /**
     * Nearest-neighbour residual variances on each side.
     *
     * @param x the running variable
     * @param y the outcome
     * @param c the cutoff
     * @param neighbours neighbours averaged, for example 3
     * @return {@code sigma_i^2}
     */
    public static double[] nnVariance(double[] x, double[] y, double c, int neighbours) {
        int n = x.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(x[a], x[b]));
        double[] out = new double[n];
        for (int k = 0; k < n; k++) {
            int i = idx[k];
            boolean above = x[i] >= c;
            double s = 0;
            int found = 0;
            int lo = k - 1;
            int hi = k + 1;
            while (found < neighbours && (lo >= 0 || hi < n)) {
                boolean useLo = lo >= 0 && (x[idx[lo]] >= c) == above;
                boolean useHi = hi < n && (x[idx[hi]] >= c) == above;
                if (!useLo && !useHi) {
                    break;
                }
                if (useLo && (!useHi || x[i] - x[idx[lo]] <= x[idx[hi]] - x[i])) {
                    s += y[idx[lo--]];
                } else {
                    s += y[idx[hi++]];
                }
                found++;
            }
            double m = found == 0 ? y[i] : s / found;
            out[i] = found / (found + 1.0) * (y[i] - m) * (y[i] - m);
        }
        return out;
    }

    private static double apply(double[] l, double[] y) {
        double s = 0;
        for (int i = 0; i < l.length; i++) {
            s += l[i] * y[i];
        }
        return s;
    }

    private static double variance(double[] l, double[] s2) {
        double v = 0;
        for (int i = 0; i < l.length; i++) {
            v += l[i] * l[i] * s2[i];
        }
        return v;
    }

    /**
     * The jump weights of the local linear fit and of its bias-corrected version.
     *
     * @return {@code {conventional, robust bias-corrected}} smoother weights for the jump
     */
    private static double[][] jumpWeights(double[] x, double c, double h, double b) {
        int n = x.length;
        double[] conv = new double[n];
        double[] robust = new double[n];
        for (boolean above : new boolean[] {true, false}) {
            double sign = above ? 1 : -1;
            double[] l = smoother(x, c, h, 1, above, 0);
            double[] k2 = smoother(x, c, b, 2, above, 2);   // estimates m''/2
            // bias constant: e1' Gamma^-1 sum w_i X_i (x_i - c)^2 = sum l_i (x_i - c)^2
            double bc = 0;
            for (int i = 0; i < n; i++) {
                bc += l[i] * (x[i] - c) * (x[i] - c);
            }
            for (int i = 0; i < n; i++) {
                conv[i] += sign * l[i];
                robust[i] += sign * (l[i] - bc * k2[i]);
            }
        }
        return new double[][] {conv, robust};
    }

    /**
     * The sharp RD estimate, conventional and robust bias-corrected.
     *
     * @param x the running variable
     * @param y the outcome
     * @param c the cutoff
     * @param h the main bandwidth
     * @param b the bias bandwidth
     * @return {@code {conventional, robust bias-corrected}}
     */
    public static Estimate[] sharp(double[] x, double[] y, double c, double h, double b) {
        double[][] w = jumpWeights(x, c, h, b);
        double[] s2 = nnVariance(x, y, c, 3);
        return new Estimate[] {Estimate.of(apply(w[0], y), Math.sqrt(variance(w[0], s2))),
            Estimate.of(apply(w[1], y), Math.sqrt(variance(w[1], s2)))};
    }

    /**
     * The fuzzy RD local Wald estimate.
     *
     * @param x the running variable
     * @param d the treatment taken (0/1)
     * @param y the outcome
     * @param c the cutoff
     * @param h the main bandwidth
     * @param b the bias bandwidth
     * @return {@code {conventional, robust bias-corrected}}
     */
    public static Estimate[] fuzzy(double[] x, int[] d, double[] y, double c, double h, double b) {
        double[][] w = jumpWeights(x, c, h, b);
        double[] dd = Arrays.stream(d).asDoubleStream().toArray();
        Estimate[] out = new Estimate[2];
        for (int k = 0; k < 2; k++) {
            double jy = apply(w[k], y);
            double jd = apply(w[k], dd);
            double tau = jy / jd;
            double[] resid = new double[y.length];
            for (int i = 0; i < y.length; i++) {
                resid[i] = y[i] - tau * dd[i];
            }
            double[] s2 = nnVariance(x, resid, c, 3);
            out[k] = Estimate.of(tau, Math.sqrt(variance(w[k], s2)) / Math.abs(jd));
        }
        return out;
    }

    private static double sideVariance(double[] x, double[] y, double c, double h, boolean above) {
        double s = 0;
        double s2 = 0;
        int k = 0;
        for (int i = 0; i < x.length; i++) {
            if ((above ? x[i] >= c : x[i] < c) && Math.abs(x[i] - c) < h) {
                s += y[i];
                s2 += y[i] * y[i];
                k++;
            }
        }
        return k < 2 ? 0 : (s2 - s * s / k) / (k - 1);
    }

    private static int count(double[] x, double c, double h, boolean above) {
        int k = 0;
        for (double v : x) {
            k += (above ? v >= c : v < c) && Math.abs(v - c) < h ? 1 : 0;
        }
        return k;
    }

    private static double sd(double[] v) {
        double m = Arrays.stream(v).average().orElse(0);
        return Math.sqrt(Arrays.stream(v).map(a -> (a - m) * (a - m)).sum() / (v.length - 1));
    }

    /**
     * The third derivative from a cubic with a jump at the cutoff, fitted (as in IK, step 2) on
     * the units between the median of the running variable below and the median above the
     * cutoff.
     */
    private static double thirdDerivative(double[] x, double[] y, double c) {
        double lo = median(Arrays.stream(x).filter(v -> v < c).toArray(), c);
        double hi = median(Arrays.stream(x).filter(v -> v >= c).toArray(), c);
        int k = 0;
        for (double v : x) {
            k += v >= lo && v <= hi ? 1 : 0;
        }
        double[][] f = new double[k][4];
        double[] t = new double[k];
        int j = 0;
        for (int i = 0; i < x.length; i++) {
            if (x[i] < lo || x[i] > hi) {
                continue;
            }
            double u = x[i] - c;
            f[j][0] = x[i] >= c ? 1 : 0;
            f[j][1] = u;
            f[j][2] = u * u;
            f[j][3] = u * u * u;
            t[j++] = y[i];
        }
        return 6 * LeastSquares.fit(f, t, true, 1e-10).beta()[4];
    }

    private static double median(double[] v, double fallback) {
        if (v.length == 0) {
            return fallback;
        }
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    /**
     * The Imbens-Kalyanaraman MSE-optimal bandwidth for the local linear sharp RD estimator with
     * a triangular kernel.
     *
     * @param x the running variable
     * @param y the outcome
     * @param c the cutoff
     * @return {@code h}
     */
    public static double ikBandwidth(double[] x, double[] y, double c) {
        int n = x.length;
        double h1 = 1.84 * sd(x) * Math.pow(n, -0.2);
        int nPlus = count(x, c, h1, true);
        int nMinus = count(x, c, h1, false);
        double f = (nPlus + nMinus) / (2.0 * n * h1);
        double vPlus = sideVariance(x, y, c, h1, true);
        double vMinus = sideVariance(x, y, c, h1, false);
        double m3 = thirdDerivative(x, y, c);
        int allPlus = count(x, c, Double.MAX_VALUE, true);
        int allMinus = count(x, c, Double.MAX_VALUE, false);
        double m3sq = Math.max(m3 * m3, 1e-8);
        double h2Plus = 3.56 * Math.pow(vPlus / (f * m3sq), 1.0 / 7) * Math.pow(allPlus, -1.0 / 7);
        double h2Minus = 3.56 * Math.pow(vMinus / (f * m3sq), 1.0 / 7) * Math.pow(allMinus, -1.0 / 7);
        double m2Plus = 2 * localQuadraticCurvature(x, y, c, h2Plus, true);
        double m2Minus = 2 * localQuadraticCurvature(x, y, c, h2Minus, false);
        int n2Plus = Math.max(1, count(x, c, h2Plus, true));
        int n2Minus = Math.max(1, count(x, c, h2Minus, false));
        double rPlus = 2160 * vPlus / (n2Plus * Math.pow(h2Plus, 4));
        double rMinus = 2160 * vMinus / (n2Minus * Math.pow(h2Minus, 4));
        double diff = m2Plus - m2Minus;
        return 3.4375 * Math.pow((vPlus + vMinus) / (f * (diff * diff + rPlus + rMinus)), 0.2) * Math.pow(n, -0.2);
    }

    /** The quadratic coefficient (m''/2) of an unweighted local quadratic on one side within h. */
    private static double localQuadraticCurvature(double[] x, double[] y, double c, double h, boolean above) {
        int k = count(x, c, h, above);
        if (k < 5) {
            return 0;
        }
        double[][] f = new double[k][2];
        double[] v = new double[k];
        int j = 0;
        for (int i = 0; i < x.length; i++) {
            if ((above ? x[i] >= c : x[i] < c) && Math.abs(x[i] - c) < h) {
                double u = x[i] - c;
                f[j][0] = u;
                f[j][1] = u * u;
                v[j++] = y[i];
            }
        }
        return LeastSquares.fit(f, v, true, 1e-12).beta()[2];
    }

    /**
     * A CCT-style plug-in pair: the bias bandwidths {@code b} are the MSE-optimal bandwidths for
     * each side's curvature (a {@code n^(-1/7)} rate from that side's third derivative, fitted by
     * a cubic up to the side's median), and {@code h} is the
     * MSE-optimal local linear bandwidth with the curvature estimated at {@code b} and without
     * IK's regularisation term.
     *
     * @param x the running variable
     * @param y the outcome
     * @param c the cutoff
     * @return {@code {h, b}}
     */
    public static double[] cctBandwidths(double[] x, double[] y, double c) {
        int n = x.length;
        double h1 = 1.84 * sd(x) * Math.pow(n, -0.2);
        double f = (count(x, c, h1, true) + count(x, c, h1, false)) / (2.0 * n * h1);
        double vPlus = sideVariance(x, y, c, h1, true);
        double vMinus = sideVariance(x, y, c, h1, false);
        // side-specific third derivatives from a cubic on each side up to that side's median
        double m3Plus = sideThirdDerivative(x, y, c, true);
        double m3Minus = sideThirdDerivative(x, y, c, false);
        int allPlus = Math.max(1, count(x, c, Double.MAX_VALUE, true));
        int allMinus = Math.max(1, count(x, c, Double.MAX_VALUE, false));
        double bPlus = 3.56 * Math.pow(vPlus / (f * Math.max(m3Plus * m3Plus, 1e-8)), 1.0 / 7)
                * Math.pow(allPlus, -1.0 / 7);
        double bMinus = 3.56 * Math.pow(vMinus / (f * Math.max(m3Minus * m3Minus, 1e-8)), 1.0 / 7)
                * Math.pow(allMinus, -1.0 / 7);
        double m2Plus = 2 * localQuadraticCurvature(x, y, c, bPlus, true);
        double m2Minus = 2 * localQuadraticCurvature(x, y, c, bMinus, false);
        double diff = m2Plus - m2Minus;
        double h = 3.4375 * Math.pow((vPlus + vMinus) / (f * Math.max(diff * diff, 1e-8)), 0.2) * Math.pow(n, -0.2);
        double range = Arrays.stream(x).map(v -> Math.abs(v - c)).max().orElse(1);
        double b = Math.max(bPlus, bMinus);
        return new double[] {Math.min(h, range), Math.min(Math.max(b, h), range)};
    }

    private static double sideThirdDerivative(double[] x, double[] y, double c, boolean above) {
        double[] side = Arrays.stream(x).filter(v -> above ? v >= c : v < c).toArray();
        double edge = median(side, c);
        int k = 0;
        for (double v : x) {
            k += above ? v >= c && v <= edge ? 1 : 0 : v < c && v >= edge ? 1 : 0;
        }
        if (k < 8) {
            return 0;
        }
        double[][] f = new double[k][3];
        double[] t = new double[k];
        int j = 0;
        for (int i = 0; i < x.length; i++) {
            boolean in = above ? x[i] >= c && x[i] <= edge : x[i] < c && x[i] >= edge;
            if (in) {
                double u = x[i] - c;
                f[j][0] = u;
                f[j][1] = u * u;
                f[j][2] = u * u * u;
                t[j++] = y[i];
            }
        }
        return 6 * LeastSquares.fit(f, t, true, 1e-12).beta()[3];
    }

    /**
     * A global polynomial fit on each side.
     *
     * @param x the running variable
     * @param y the outcome
     * @param c the cutoff
     * @param order the polynomial order
     * @return the jump of the two intercepts
     */
    public static double globalPolynomial(double[] x, double[] y, double c, int order) {
        double jump = 0;
        for (boolean above : new boolean[] {true, false}) {
            int k = 0;
            for (double v : x) {
                k += (above ? v >= c : v < c) ? 1 : 0;
            }
            double[][] f = new double[k][order];
            double[] v = new double[k];
            int j = 0;
            for (int i = 0; i < x.length; i++) {
                if (above ? x[i] >= c : x[i] < c) {
                    double u = x[i] - c;
                    double q = u;
                    for (int a = 0; a < order; a++) {
                        f[j][a] = q;
                        q *= u;
                    }
                    v[j++] = y[i];
                }
            }
            jump += (above ? 1 : -1) * LeastSquares.fit(f, v, true, 1e-12).beta()[0];
        }
        return jump;
    }

    /**
     * The difference of means in a fixed window around the cutoff.
     *
     * @param x the running variable
     * @param y the outcome
     * @param c the cutoff
     * @param width the half-width
     * @return {@code mean(y | c <= x < c + w) - mean(y | c - w < x < c)}
     */
    public static double windowMeans(double[] x, double[] y, double c, double width) {
        double sp = 0;
        double sm = 0;
        int np = 0;
        int nm = 0;
        for (int i = 0; i < x.length; i++) {
            if (x[i] >= c && x[i] < c + width) {
                sp += y[i];
                np++;
            } else if (x[i] < c && x[i] > c - width) {
                sm += y[i];
                nm++;
            }
        }
        return sp / Math.max(1, np) - sm / Math.max(1, nm);
    }

    /**
     * McCrary's density test.
     *
     * @param x the running variable
     * @param c the cutoff
     * @return {@code {log density jump theta, its standard error, two-sided p-value}}
     */
    public static double[] densityTest(double[] x, double c) {
        int n = x.length;
        double sx = sd(x);
        double bin = 2 * sx / Math.sqrt(n);
        double lo = Arrays.stream(x).min().orElse(c);
        double hi = Arrays.stream(x).max().orElse(c);
        int left = (int) Math.ceil((c - lo) / bin);
        int right = (int) Math.ceil((hi - c) / bin);
        double[] mids = new double[left + right];
        double[] dens = new double[left + right];
        for (int k = 0; k < left + right; k++) {
            mids[k] = c + (k - left + 0.5) * bin;
        }
        for (double v : x) {
            int k = (int) Math.floor((v - c) / bin) + left;
            if (k >= 0 && k < dens.length) {
                dens[k] += 1 / (n * bin);
            }
        }
        double h = 2 * sx * Math.pow(n, -0.2);
        double[] f = new double[2];
        for (int s = 0; s < 2; s++) {
            boolean above = s == 0;
            double[] l = smoother(mids, c, h, 1, above, 0);
            f[s] = Math.max(apply(l, dens), 1e-12);
        }
        double theta = Math.log(f[0]) - Math.log(f[1]);
        double se = Math.sqrt(1 / (n * h) * 24.0 / 5 * (1 / f[0] + 1 / f[1]));
        return new double[] {theta, se, 2 * (1 - NormalTails.cdf(Math.abs(theta / se)))};
    }
}
