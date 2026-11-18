package io.github.zzuegg.brainiac.experimental.discovery;

import io.github.zzuegg.brainiac.experimental.ml.KernelTests;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.Arrays;

/**
 * Bivariate cause-effect inference, where conditional independence says nothing about the
 * direction. Every method returns a score that is positive when it prefers {@code X -> Y}; its
 * magnitude is the confidence used for accuracy-decision-rate curves.
 *
 * <ul>
 *   <li>{@link #anm}: additive noise models (Hoyer et al., NeurIPS 2008). Kernel ridge regression
 *       both ways, and the direction whose residuals are more independent of the input (smaller
 *       HSIC) wins.</li>
 *   <li>{@link #igciSlope} and {@link #igciEntropy}: information-geometric causal inference
 *       (Janzing et al., AIJ 2012) with a uniform reference measure (both variables rescaled to
 *       {@code [0, 1]}): the slope estimator {@code C = mean log |dy/dx|} and the entropy
 *       estimator {@code C = H(Y) - H(X)} (m-spacing entropies), preferring the direction with
 *       the smaller {@code C}; meant for near-deterministic mechanisms.</li>
 *   <li>{@link #reci}: regression error based causal inference (Blöbaum et al., AISTATS 2018):
 *       after min-max scaling, the direction with the smaller mean squared error of a cubic
 *       polynomial regression.</li>
 *   <li>Baselines: {@link #naiveHsic}, the ANM test with a linear regression (dependence of the
 *       residual on the input, so it only uses non-Gaussianity), and {@link #lingam}, the
 *       DirectLiNGAM pairwise likelihood ratio.</li>
 * </ul>
 */
public final class CauseEffect {

    private CauseEffect() {
    }

    private static double[] standard(double[] v) {
        double m = Arrays.stream(v).average().orElse(0);
        double s = Math.sqrt(Arrays.stream(v).map(a -> (a - m) * (a - m)).average().orElse(1));
        double[] o = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            o[i] = (v[i] - m) / Math.max(s, 1e-12);
        }
        return o;
    }

    private static double[] unit(double[] v) {
        double lo = Arrays.stream(v).min().orElse(0);
        double hi = Arrays.stream(v).max().orElse(1);
        double[] o = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            o[i] = (v[i] - lo) / Math.max(hi - lo, 1e-12);
        }
        return o;
    }

    private static double hsic(double[] a, double[] b) {
        double[][] x = new double[a.length][1];
        double[][] y = new double[a.length][1];
        for (int i = 0; i < a.length; i++) {
            x[i][0] = a[i];
            y[i][0] = b[i];
        }
        return KernelTests.hsicTest(x, y, 0, 0).statistic();
    }

    /**
     * Kernel ridge regression residuals of {@code y} on {@code x} (Gaussian kernel, median
     * bandwidth).
     *
     * @param x the input
     * @param y the output
     * @param ridge the regulariser
     * @return {@code y - f(x)}
     */
    public static double[] kernelRidgeResiduals(double[] x, double[] y, double ridge) {
        int n = x.length;
        double[][] xs = new double[n][1];
        for (int i = 0; i < n; i++) {
            xs[i][0] = x[i];
        }
        double sigma = KernelTests.medianDistance(xs);
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dd = x[i] - x[j];
                k[i][j] = Math.exp(-dd * dd / (2 * sigma * sigma));
            }
        }
        double[][] m = new double[n][];
        for (int i = 0; i < n; i++) {
            m[i] = k[i].clone();
            m[i][i] += ridge;
        }
        double[] alpha = MlMatrices.choleskySolve(MlMatrices.cholesky(m), y);
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            r[i] = y[i] - MlMatrices.dot(k[i], alpha);
        }
        return r;
    }

    /**
     * The additive-noise-model score.
     *
     * @param x one variable
     * @param y the other
     * @return {@code HSIC(y, r_{x|y}) - HSIC(x, r_{y|x})}
     */
    public static double anm(double[] x, double[] y) {
        double[] a = standard(x);
        double[] b = standard(y);
        double ridge = 0.05;
        double forward = hsic(a, kernelRidgeResiduals(a, b, ridge));
        double backward = hsic(b, kernelRidgeResiduals(b, a, ridge));
        return backward - forward;
    }

    /**
     * The linear-regression ANM baseline.
     *
     * @param x one variable
     * @param y the other
     * @return {@code HSIC(y, linear residual of x) - HSIC(x, linear residual of y)}
     */
    public static double naiveHsic(double[] x, double[] y) {
        double[] a = standard(x);
        double[] b = standard(y);
        double c = 0;
        for (int i = 0; i < a.length; i++) {
            c += a[i] * b[i] / a.length;
        }
        double[] ry = new double[a.length];
        double[] rx = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ry[i] = b[i] - c * a[i];
            rx[i] = a[i] - c * b[i];
        }
        return hsic(b, rx) - hsic(a, ry);
    }

    /**
     * The DirectLiNGAM pairwise measure as a direction score.
     *
     * @param x one variable
     * @param y the other
     * @return positive for {@code X -> Y}
     */
    public static double lingam(double[] x, double[] y) {
        return Lingam.pairwiseMeasure(standard(x), standard(y));
    }

    private static double slope(double[] x, double[] y) {
        int n = x.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(x[p], x[q]));
        double s = 0;
        int c = 0;
        for (int k = 0; k + 1 < n; k++) {
            double dx = x[idx[k + 1]] - x[idx[k]];
            double dy = y[idx[k + 1]] - y[idx[k]];
            if (dx != 0 && dy != 0) {
                s += Math.log(Math.abs(dy / dx));
                c++;
            }
        }
        return c == 0 ? 0 : s / c;
    }

    /**
     * IGCI with the slope estimator.
     *
     * @param x one variable
     * @param y the other
     * @return {@code C_{Y->X} - C_{X->Y}}
     */
    public static double igciSlope(double[] x, double[] y) {
        double[] a = unit(x);
        double[] b = unit(y);
        return slope(b, a) - slope(a, b);
    }

    /**
     * The m-spacing (Vasicek) entropy estimate.
     *
     * @param v samples
     * @return the differential entropy estimate
     */
    public static double spacingEntropy(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        int n = s.length;
        int m = Math.max(1, (int) Math.round(Math.sqrt(n)));
        double h = 0;
        for (int i = 0; i < n; i++) {
            double gap = s[Math.min(n - 1, i + m)] - s[Math.max(0, i - m)];
            h += Math.log(n / (2.0 * m) * Math.max(gap, 1e-12));
        }
        return h / n;
    }

    /**
     * IGCI with the entropy estimator.
     *
     * @param x one variable
     * @param y the other
     * @return {@code H(X) - H(Y)} after rescaling to {@code [0, 1]}
     */
    public static double igciEntropy(double[] x, double[] y) {
        return spacingEntropy(unit(x)) - spacingEntropy(unit(y));
    }

    private static double polyMse(double[] x, double[] y) {
        double[][] f = new double[x.length][3];
        for (int i = 0; i < x.length; i++) {
            f[i][0] = x[i];
            f[i][1] = x[i] * x[i];
            f[i][2] = x[i] * x[i] * x[i];
        }
        return LeastSquares.fit(f, y, true, 1e-9).rss() / x.length;
    }

    /**
     * RECI.
     *
     * @param x one variable
     * @param y the other
     * @return {@code MSE(X | Y) - MSE(Y | X)} after min-max scaling
     */
    public static double reci(double[] x, double[] y) {
        double[] a = unit(x);
        double[] b = unit(y);
        return polyMse(b, a) - polyMse(a, b);
    }

    /**
     * The accuracy and area under the accuracy-decision-rate curve of direction scores.
     *
     * @param scores per pair, positive meaning {@code X -> Y}
     * @param truth per pair, whether {@code X -> Y} is true
     * @return {@code {accuracy, AUC of accuracy against decision rate}}
     */
    public static double[] accuracyCurve(double[] scores, boolean[] truth) {
        int n = scores.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(Math.abs(scores[q]), Math.abs(scores[p])));
        double correct = 0;
        double area = 0;
        for (int k = 0; k < n; k++) {
            int i = idx[k];
            double ok = scores[i] == 0 ? 0.5 : (scores[i] > 0) == truth[i] ? 1 : 0;
            correct += ok;
            area += correct / (k + 1) / n;
        }
        return new double[] {correct / n, area};
    }
}
