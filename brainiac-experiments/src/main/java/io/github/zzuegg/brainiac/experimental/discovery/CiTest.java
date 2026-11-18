package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

/**
 * A conditional-independence test over the columns of one data set, the only thing the
 * constraint-based learners ({@link PcAlgorithm}, {@link Fci}) need.
 *
 * <ul>
 *   <li>{@link #fisherZ}: the partial correlation from the inverse of the correlation submatrix,
 *       Fisher-z transformed, {@code z = atanh(r) sqrt(n - |S| - 3)}; exact for Gaussian data.</li>
 *   <li>{@link #kernel}: both variables are residualised on the conditioning set by kernel ridge
 *       regression (Gaussian kernel, median bandwidth), then the residuals are tested by HSIC with
 *       the gamma approximation of {@link KernelTests#hsicTest}; a regression-based kernel CI test
 *       that catches nonlinear dependence (a cheaper relative of KCIT).</li>
 * </ul>
 */
@FunctionalInterface
public interface CiTest {

    /**
     * The p-value of {@code X_i independent of X_j given X_S}.
     *
     * @param i one variable
     * @param j the other
     * @param s the conditioning set
     * @return the p-value; large means independent
     */
    double pValue(int i, int j, int[] s);

    /**
     * The Fisher-z partial-correlation test.
     *
     * @param data the samples {@code [n][d]}
     * @return the test
     */
    static CiTest fisherZ(double[][] data) {
        int n = data.length;
        int d = data[0].length;
        double[][] corr = new double[d][d];
        double[][] cols = new double[d][];
        for (int j = 0; j < d; j++) {
            cols[j] = LeastSquares.column(data, j);
        }
        for (int a = 0; a < d; a++) {
            for (int b = a; b < d; b++) {
                corr[a][b] = a == b ? 1 : LeastSquares.correlation(cols[a], cols[b]);
                corr[b][a] = corr[a][b];
            }
        }
        return (i, j, s) -> {
            double r = partialCorrelation(corr, i, j, s);
            r = Math.max(-0.999999, Math.min(0.999999, r));
            double z = 0.5 * Math.log((1 + r) / (1 - r)) * Math.sqrt(Math.max(1, n - s.length - 3));
            return Distributions.normalTwoSided(z);
        };
    }

    /**
     * The partial correlation of {@code i} and {@code j} given {@code s} from a correlation
     * matrix.
     *
     * @param corr the correlation matrix
     * @param i one variable
     * @param j the other
     * @param s the conditioning set
     * @return the partial correlation
     */
    static double partialCorrelation(double[][] corr, int i, int j, int[] s) {
        if (s.length == 0) {
            return corr[i][j];
        }
        int m = s.length + 2;
        int[] idx = new int[m];
        idx[0] = i;
        idx[1] = j;
        System.arraycopy(s, 0, idx, 2, s.length);
        double[][] sub = new double[m][m];
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                sub[a][b] = corr[idx[a]][idx[b]];
            }
        }
        double[][] p = MlMatrices.choleskyInverse(MlMatrices.cholesky(sub));
        return -p[0][1] / Math.sqrt(p[0][0] * p[1][1]);
    }

    /**
     * The residual-HSIC kernel test.
     *
     * @param data the samples {@code [n][d]}
     * @param maxRows at most this many rows are used (the first ones), for speed
     * @param ridge the kernel-ridge regulariser, for example {@code 0.1}
     * @return the test
     */
    static CiTest kernel(double[][] data, int maxRows, double ridge) {
        int n = Math.min(maxRows, data.length);
        double[][] x = LeastSquares.standardise(java.util.Arrays.copyOf(data, n));
        return (i, j, s) -> {
            double[] ri = LeastSquares.column(x, i);
            double[] rj = LeastSquares.column(x, j);
            if (s.length > 0) {
                double[][] z = LeastSquares.columns(x, s);
                double sigma = KernelTests.medianDistance(z);
                double[][] k = new double[n][n];
                for (int a = 0; a < n; a++) {
                    for (int b = a; b < n; b++) {
                        k[a][b] = Math.exp(-MlMatrices.squaredDistance(z[a], z[b]) / (2 * sigma * sigma));
                        k[b][a] = k[a][b];
                    }
                    k[a][a] += ridge;
                }
                double[][] l = MlMatrices.cholesky(k);
                ri = residual(k, l, ri, ridge);
                rj = residual(k, l, rj, ridge);
            }
            double[][] a = new double[n][1];
            double[][] b = new double[n][1];
            for (int t = 0; t < n; t++) {
                a[t][0] = ri[t];
                b[t][0] = rj[t];
            }
            return KernelTests.hsicTest(a, b, 0, 0).pValue();
        };
    }

    private static double[] residual(double[][] kRidge, double[][] chol, double[] y, double ridge) {
        double[] alpha = MlMatrices.choleskySolve(chol, y);
        double[] out = new double[y.length];
        for (int a = 0; a < y.length; a++) {
            double f = 0;
            for (int b = 0; b < y.length; b++) {
                f += (kRidge[a][b] - (a == b ? ridge : 0)) * alpha[b];
            }
            out[a] = y[a] - f;
        }
        return out;
    }
}
