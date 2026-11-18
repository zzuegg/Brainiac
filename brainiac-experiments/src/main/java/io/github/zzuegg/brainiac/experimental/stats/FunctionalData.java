package io.github.zzuegg.brainiac.experimental.stats;

import java.util.Arrays;

import io.github.zzuegg.brainiac.tensor.Matrices;

/**
 * Functional data analysis on a common evaluation grid.
 * <ul>
 *   <li>{@link #smooth}: penalised B-spline smoothing of one noisy curve
 *       ({@link PenalizedSplines}, GCV-chosen penalty);</li>
 *   <li>{@link #fpca}: functional principal components of densely observed curves (eigen
 *       decomposition of the sample covariance on the grid, scaled by the grid step);</li>
 *   <li>{@link #pace}: PACE for sparse, irregular curves (Yao, Mueller and Wang, JASA 2005):
 *       the mean is smoothed from pooled points, the covariance from pooled off-diagonal raw
 *       products binned onto the grid and kernel-smoothed, the noise variance from the diagonal,
 *       and scores are the conditional expectations (BLUP)
 *       {@code xi_i = Lambda Phi_i^T Sigma_i^-1 (Y_i - mu_i)};</li>
 *   <li>{@link #regression}: scalar-on-function regression in the FPC basis;</li>
 *   <li>{@link #modifiedBandDepth} (Lopez-Pintado and Romo, JASA 2009),
 *       {@link #boxplotOutliers} (Sun and Genton, JCGS 2011: the central 50% region by depth,
 *       inflated 1.5 times) and {@link #outliergram} (Arribas-Gil and Romo, Biostatistics 2014:
 *       the distance below the parabola linking band depth and the modified epigraph index,
 *       which flags shape outliers).</li>
 * </ul>
 */
public final class FunctionalData {

    private FunctionalData() {
    }

    /**
     * A functional principal component basis.
     *
     * @param grid the evaluation grid
     * @param mean the mean function on the grid
     * @param components eigenfunctions {@code [k][grid]}, orthonormal in {@code L2}
     * @param eigenvalues their variances
     * @param noise the estimated measurement-noise variance (0 for dense FPCA)
     */
    public record Basis(double[] grid, double[] mean, double[][] components, double[] eigenvalues,
            double noise) {

        /**
         * Reconstructs curves from scores.
         *
         * @param scores {@code [n][k]}
         * @return curves {@code [n][grid]}
         */
        public double[][] reconstruct(double[][] scores) {
            double[][] out = new double[scores.length][grid.length];
            for (int i = 0; i < scores.length; i++) {
                for (int g = 0; g < grid.length; g++) {
                    double v = mean[g];
                    for (int c = 0; c < components.length; c++) {
                        v += scores[i][c] * components[c][g];
                    }
                    out[i][g] = v;
                }
            }
            return out;
        }

        /**
         * Scores of densely observed curves by numerical integration.
         *
         * @param curves {@code [n][grid]}
         * @return scores {@code [n][k]}
         */
        public double[][] project(double[][] curves) {
            double h = step(grid);
            double[][] out = new double[curves.length][components.length];
            for (int i = 0; i < curves.length; i++) {
                for (int c = 0; c < components.length; c++) {
                    double s = 0;
                    for (int g = 0; g < grid.length; g++) {
                        s += (curves[i][g] - mean[g]) * components[c][g] * h;
                    }
                    out[i][c] = s;
                }
            }
            return out;
        }

        /**
         * BLUP (PACE) scores of sparsely observed curves.
         *
         * @param times observation times per curve
         * @param values observed values per curve
         * @return scores {@code [n][k]}
         */
        public double[][] blup(double[][] times, double[][] values) {
            int k = components.length;
            double[][] out = new double[times.length][];
            for (int i = 0; i < times.length; i++) {
                int m = times[i].length;
                double[][] phi = new double[m][k];
                double[] r = new double[m];
                for (int j = 0; j < m; j++) {
                    for (int c = 0; c < k; c++) {
                        phi[j][c] = interpolate(grid, components[c], times[i][j]);
                    }
                    r[j] = values[i][j] - interpolate(grid, mean, times[i][j]);
                }
                double[][] sigma = new double[m][m];
                for (int a = 0; a < m; a++) {
                    for (int b = 0; b < m; b++) {
                        double s = a == b ? Math.max(noise, 1e-6) : 0;
                        for (int c = 0; c < k; c++) {
                            s += phi[a][c] * eigenvalues[c] * phi[b][c];
                        }
                        sigma[a][b] = s;
                    }
                }
                double[] z = Matrices.solve(sigma, r);
                double[] xi = new double[k];
                for (int c = 0; c < k; c++) {
                    double s = 0;
                    for (int j = 0; j < m; j++) {
                        s += phi[j][c] * z[j];
                    }
                    xi[c] = eigenvalues[c] * s;
                }
                out[i] = xi;
            }
            return out;
        }
    }

    private static double step(double[] grid) {
        return (grid[grid.length - 1] - grid[0]) / (grid.length - 1);
    }

    /**
     * Linear interpolation on an ascending grid, clamped at the ends.
     *
     * @param grid the grid
     * @param f values on the grid
     * @param t the point
     * @return {@code f(t)}
     */
    public static double interpolate(double[] grid, double[] f, double t) {
        if (t <= grid[0]) {
            return f[0];
        }
        int last = grid.length - 1;
        if (t >= grid[last]) {
            return f[last];
        }
        int j = Arrays.binarySearch(grid, t);
        if (j >= 0) {
            return f[j];
        }
        int hi = -j - 1;
        int lo = hi - 1;
        double w = (t - grid[lo]) / (grid[hi] - grid[lo]);
        return (1 - w) * f[lo] + w * f[hi];
    }

    /**
     * Smooths one noisy curve onto a grid.
     *
     * @param t observation times
     * @param y observations
     * @param grid the evaluation grid
     * @param segments B-spline segments
     * @return the smoothed curve on the grid
     */
    public static double[] smooth(double[] t, double[] y, double[] grid, int segments) {
        PenalizedSplines s = PenalizedSplines.fit(t, y, null, segments, Double.NaN);
        double[] out = new double[grid.length];
        for (int g = 0; g < grid.length; g++) {
            out[g] = s.predict(grid[g]);
        }
        return out;
    }

    private static Basis fromCovariance(double[] grid, double[] mean, double[][] cov, int k, double noise) {
        int m = grid.length;
        double h = step(grid);
        double[][] scaled = new double[m][m];
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                scaled[a][b] = cov[a][b] * h;
            }
        }
        Matrices.SymmetricEigen e = Matrices.symmetricEigen(scaled);
        double[][] comp = new double[k][m];
        double[] lam = new double[k];
        for (int c = 0; c < k; c++) {
            int idx = m - 1 - c;
            lam[c] = Math.max(e.values()[idx], 1e-12);
            double sign = 0;
            for (int g = 0; g < m; g++) {
                comp[c][g] = e.vectors()[g][idx] / Math.sqrt(h);
                sign += comp[c][g];
            }
            if (sign < 0) {
                for (int g = 0; g < m; g++) {
                    comp[c][g] = -comp[c][g];
                }
            }
        }
        return new Basis(grid.clone(), mean, comp, lam, noise);
    }

    /**
     * Dense FPCA.
     *
     * @param grid the common grid
     * @param curves {@code [n][grid]}
     * @param k the number of components
     * @return the basis
     */
    public static Basis fpca(double[] grid, double[][] curves, int k) {
        int n = curves.length;
        int m = grid.length;
        double[] mean = new double[m];
        for (double[] c : curves) {
            for (int g = 0; g < m; g++) {
                mean[g] += c[g] / n;
            }
        }
        double[][] cov = new double[m][m];
        for (double[] c : curves) {
            for (int a = 0; a < m; a++) {
                for (int b = 0; b < m; b++) {
                    cov[a][b] += (c[a] - mean[a]) * (c[b] - mean[b]) / (n - 1);
                }
            }
        }
        return fromCovariance(grid, mean, cov, k, 0);
    }

    /**
     * PACE for sparse curves.
     *
     * @param times observation times per curve (within the grid range)
     * @param values observations per curve
     * @param grid the evaluation grid
     * @param k the number of components
     * @param segments B-spline segments for the mean
     * @param bandwidth the Gaussian kernel bandwidth for the covariance surface
     * @return the basis, including the noise variance
     */
    public static Basis pace(double[][] times, double[][] values, double[] grid, int k, int segments,
            double bandwidth) {
        int total = 0;
        for (double[] t : times) {
            total += t.length;
        }
        double[] pt = new double[total];
        double[] py = new double[total];
        int p = 0;
        for (int i = 0; i < times.length; i++) {
            for (int j = 0; j < times[i].length; j++) {
                pt[p] = times[i][j];
                py[p++] = values[i][j];
            }
        }
        double[] mean = smooth(pt, py, grid, segments);
        int m = grid.length;
        double[][] sum = new double[m][m];
        double[][] cnt = new double[m][m];
        double[] diagSum = new double[m];
        double[] diagCnt = new double[m];
        for (int i = 0; i < times.length; i++) {
            int len = times[i].length;
            int[] bin = new int[len];
            double[] r = new double[len];
            for (int j = 0; j < len; j++) {
                bin[j] = nearest(grid, times[i][j]);
                r[j] = values[i][j] - interpolate(grid, mean, times[i][j]);
            }
            for (int a = 0; a < len; a++) {
                diagSum[bin[a]] += r[a] * r[a];
                diagCnt[bin[a]]++;
                for (int b = 0; b < len; b++) {
                    if (a != b) {
                        sum[bin[a]][bin[b]] += r[a] * r[b];
                        cnt[bin[a]][bin[b]]++;
                    }
                }
            }
        }
        double[][] cov = new double[m][m];
        double[][] w1 = new double[m][m];
        for (int a = 0; a < m; a++) {
            for (int c = 0; c < m; c++) {
                double d = (grid[a] - grid[c]) / bandwidth;
                w1[a][c] = Math.exp(-0.5 * d * d);
            }
        }
        // separable kernel smoothing: num = W S W, den = W C W
        double[][] num = Matrices.multiply(Matrices.multiply(w1, sum), w1);
        double[][] den = Matrices.multiply(Matrices.multiply(w1, cnt), w1);
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                cov[a][b] = den[a][b] > 1e-12 ? num[a][b] / den[a][b] : 0;
            }
        }
        for (int a = 0; a < m; a++) {
            for (int b = a + 1; b < m; b++) {
                double s = 0.5 * (cov[a][b] + cov[b][a]);
                cov[a][b] = s;
                cov[b][a] = s;
            }
        }
        // the noise: smoothed raw diagonal minus the covariance surface diagonal, on the middle half
        double noise = 0;
        int used = 0;
        for (int a = m / 4; a < 3 * m / 4; a++) {
            double ns = 0;
            double nc = 0;
            for (int c = 0; c < m; c++) {
                ns += w1[a][c] * diagSum[c];
                nc += w1[a][c] * diagCnt[c];
            }
            if (nc > 1e-12) {
                noise += ns / nc - cov[a][a];
                used++;
            }
        }
        noise = used > 0 ? Math.max(noise / used, 1e-6) : 1e-6;
        return fromCovariance(grid, mean, cov, k, noise);
    }

    private static int nearest(double[] grid, double t) {
        int best = 0;
        for (int g = 1; g < grid.length; g++) {
            if (Math.abs(grid[g] - t) < Math.abs(grid[best] - t)) {
                best = g;
            }
        }
        return best;
    }

    /**
     * Scalar-on-function regression in the FPC basis: least squares on the scores plus an
     * intercept.
     *
     * @param scores {@code [n][k]}
     * @param y responses
     * @return coefficients {@code {intercept, b_1..b_k}}
     */
    public static double[] regression(double[][] scores, double[] y) {
        int n = y.length;
        int k = scores[0].length + 1;
        double[][] xtx = new double[k][k];
        double[] xty = new double[k];
        for (int i = 0; i < n; i++) {
            double[] row = new double[k];
            row[0] = 1;
            System.arraycopy(scores[i], 0, row, 1, k - 1);
            for (int a = 0; a < k; a++) {
                xty[a] += row[a] * y[i];
                for (int b = 0; b < k; b++) {
                    xtx[a][b] += row[a] * row[b];
                }
            }
        }
        for (int a = 1; a < k; a++) {
            xtx[a][a] += 1e-8;
        }
        return Matrices.solve(xtx, xty);
    }

    /**
     * Predictions of a scores regression.
     *
     * @param coefficients from {@link #regression}
     * @param scores {@code [n][k]}
     * @return predictions
     */
    public static double[] predict(double[] coefficients, double[][] scores) {
        double[] out = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            double v = coefficients[0];
            for (int c = 0; c < scores[i].length; c++) {
                v += coefficients[c + 1] * scores[i][c];
            }
            out[i] = v;
        }
        return out;
    }

    /**
     * Modified band depth with bands of two curves.
     *
     * @param curves {@code [n][grid]}
     * @return per-curve depth in {@code [0, 1]}
     */
    public static double[] modifiedBandDepth(double[][] curves) {
        int n = curves.length;
        int m = curves[0].length;
        double pairs = n * (n - 1) / 2.0;
        double[] depth = new double[n];
        for (int g = 0; g < m; g++) {
            for (int i = 0; i < n; i++) {
                int below = 0;
                int above = 0;
                for (int j = 0; j < n; j++) {
                    if (curves[j][g] < curves[i][g]) {
                        below++;
                    } else if (curves[j][g] > curves[i][g]) {
                        above++;
                    }
                }
                double outside = below * (below - 1) / 2.0 + above * (above - 1) / 2.0;
                depth[i] += (pairs - outside) / pairs / m;
            }
        }
        return depth;
    }

    /**
     * The modified epigraph index: the mean proportion of curves lying above each curve.
     *
     * @param curves {@code [n][grid]}
     * @return per-curve index
     */
    public static double[] modifiedEpigraphIndex(double[][] curves) {
        int n = curves.length;
        int m = curves[0].length;
        double[] mei = new double[n];
        for (int g = 0; g < m; g++) {
            for (int i = 0; i < n; i++) {
                int above = 0;
                for (int j = 0; j < n; j++) {
                    if (curves[j][g] >= curves[i][g]) {
                        above++;
                    }
                }
                mei[i] += above / (double) n / m;
            }
        }
        return mei;
    }

    /**
     * Functional-boxplot magnitude outlyingness: how far each curve leaves the 1.5-times
     * inflated envelope of the deepest half, as the maximum excursion relative to the envelope
     * width (0 inside).
     *
     * @param curves {@code [n][grid]}
     * @return per-curve outlyingness ({@code > 0} flags an outlier)
     */
    public static double[] boxplotOutliers(double[][] curves) {
        int n = curves.length;
        int m = curves[0].length;
        double[] depth = modifiedBandDepth(curves);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(depth[b], depth[a]));
        int half = Math.max(1, (int) Math.ceil(n / 2.0));
        double[] out = new double[n];
        for (int g = 0; g < m; g++) {
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (int r = 0; r < half; r++) {
                lo = Math.min(lo, curves[order[r]][g]);
                hi = Math.max(hi, curves[order[r]][g]);
            }
            double width = Math.max(hi - lo, 1e-12);
            double flo = lo - 1.5 * width;
            double fhi = hi + 1.5 * width;
            for (int i = 0; i < n; i++) {
                double e = Math.max(flo - curves[i][g], curves[i][g] - fhi);
                out[i] = Math.max(out[i], Math.max(e, 0) / width);
            }
        }
        return out;
    }

    /**
     * The outliergram distance: how far each curve's band depth falls below the parabola
     * {@code a0 + a1 MEI + a2 n^2 MEI^2} that bounds it, with
     * {@code a0 = a2 = -2/(n(n-1))} and {@code a1 = 2(n+1)/(n-1)}. Large values flag shape
     * outliers.
     *
     * @param curves {@code [n][grid]}
     * @return per-curve distance
     */
    public static double[] outliergram(double[][] curves) {
        int n = curves.length;
        double[] mbd = modifiedBandDepth(curves);
        double[] mei = modifiedEpigraphIndex(curves);
        double a0 = -2.0 / (n * (n - 1.0));
        double a1 = 2.0 * (n + 1) / (n - 1.0);
        double[] d = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = a0 + a1 * mei[i] + a0 * n * n * mei[i] * mei[i] - mbd[i];
        }
        return d;
    }
}
