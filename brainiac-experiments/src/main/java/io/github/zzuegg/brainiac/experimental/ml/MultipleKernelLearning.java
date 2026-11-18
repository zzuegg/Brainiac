package io.github.zzuegg.brainiac.experimental.ml;

import java.util.List;

/**
 * Multiple kernel learning for binary classification: a non-negative combination
 * {@code K = sum_m mu_m K_m} of base kernels (bandwidths, or kernels on feature groups), each base
 * kernel scaled to unit mean diagonal.
 * <ul>
 *   <li>{@link Method#SIMPLE_MKL} (Rakotomamonjy et al., JMLR 2008): minimises the SVM dual optimum
 *       {@code J(mu) = sum_i alpha_i - 1/2 sum_ij alpha_i alpha_j y_i y_j K_mu(i, j)} over the simplex
 *       by projected gradient steps with a backtracking line search; the gradient
 *       {@code dJ / dmu_m = -1/2 w^T K_m w} ({@code w = alpha y}) comes from the {@link KernelSvm}
 *       solution. {@code KernelSvm} keeps its dual coefficients private, so the SVM is trained on
 *       index tokens with a look-up kernel and {@code w} is read back by probing
 *       {@link KernelSvm#decision} with unit tokens.</li>
 *   <li>{@link Method#EASY_MKL} (Aiolli and Donini, Neurocomputing 2015): solves
 *       {@code min_gamma (1 - lambda) gamma^T Y K Y gamma + lambda |gamma|^2} over the product of the two
 *       class simplices for the average kernel, sets {@code mu_m ~ gamma^T Y K_m Y gamma}, solves again
 *       for the combined kernel and classifies by the centroid margin. Both QPs are solved by
 *       accelerated projected gradient; no SVM is trained.</li>
 *   <li>{@link Method#ALIGNF} (Cortes, Mohri and Rostamizadeh, JMLR 2012): maximises the centred
 *       alignment of the combination with {@code y y^T}: {@code min_v>=0 v^T M v - 2 v^T a} with
 *       {@code M_kl = <K_k^c, K_l^c>_F} and {@code a_k = y^T K_k^c y}, solved as a non-negative least
 *       squares problem by accelerated projected gradient (no QP solver in {@code Optimize}); then
 *       an SVM on the combined kernel.</li>
 *   <li>{@link Method#UNIFORM}: the plain average, then an SVM.</li>
 * </ul>
 */
public final class MultipleKernelLearning {

    /** The weight learner. */
    public enum Method {
        /** The uniform average of the base kernels. */
        UNIFORM,
        /** SimpleMKL: reduced-gradient descent of the SVM dual on the simplex. */
        SIMPLE_MKL,
        /** EasyMKL: centroid-margin weights in closed form. */
        EASY_MKL,
        /** ALIGNF: centred kernel-target alignment. */
        ALIGNF
    }

    private final Method method;
    private final List<KernelFunction> kernels;
    private final double[] scales;
    private final double[] mu;
    private final double[][] x;
    private final KernelSvm svm;
    private final double[] coefficients;
    private final double bias;
    private final int svmSolves;

    private MultipleKernelLearning(Method method, List<KernelFunction> kernels, double[] scales, double[] mu,
            double[][] x, KernelSvm svm, double[] coefficients, double bias, int svmSolves) {
        this.method = method;
        this.kernels = kernels;
        this.scales = scales;
        this.mu = mu;
        this.x = x;
        this.svm = svm;
        this.coefficients = coefficients;
        this.bias = bias;
        this.svmSolves = svmSolves;
    }

    /**
     * Fits.
     *
     * @param method the weight learner
     * @param x the training points {@code [n][d]}
     * @param labels binary labels {@code 0} or {@code 1}
     * @param kernels the base kernels
     * @param c the SVM box constraint
     * @param lambda the EasyMKL regulariser in {@code (0, 1)}, such as 0.1 (ignored by the others)
     * @param iterations SimpleMKL outer iterations, such as 20 (ignored by the others)
     * @return the model
     */
    public static MultipleKernelLearning fit(Method method, double[][] x, int[] labels, List<KernelFunction> kernels,
            double c, double lambda, int iterations) {
        int n = x.length;
        int m = kernels.size();
        double[][][] grams = new double[m][][];
        double[] scales = new double[m];
        for (int k = 0; k < m; k++) {
            double[][] g = kernels.get(k).gram(x);
            double diag = 0;
            for (int i = 0; i < n; i++) {
                diag += g[i][i] / n;
            }
            scales[k] = 1 / Math.max(1e-12, diag);
            for (double[] row : g) {
                for (int j = 0; j < n; j++) {
                    row[j] *= scales[k];
                }
            }
            grams[k] = g;
        }
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = labels[i] == 1 ? 1 : -1;
        }
        double[] mu = new double[m];
        java.util.Arrays.fill(mu, 1.0 / m);
        int[] solves = {0};
        switch (method) {
            case UNIFORM -> {
            }
            case ALIGNF -> mu = alignF(grams, y);
            case SIMPLE_MKL -> mu = simpleMkl(grams, labels, mu, c, iterations, solves);
            case EASY_MKL -> {
                double[][] avg = combine(grams, mu);
                double[] gamma = easyGamma(avg, y, lambda);
                double[] eta = new double[m];
                double total = 0;
                for (int k = 0; k < m; k++) {
                    eta[k] = quadratic(grams[k], gamma, y);
                    total += eta[k];
                }
                for (int k = 0; k < m; k++) {
                    mu[k] = eta[k] / Math.max(1e-300, total);
                }
                double[][] combined = combine(grams, mu);
                double[] g2 = easyGamma(combined, y, lambda);
                double[] coef = new double[n];
                for (int i = 0; i < n; i++) {
                    coef[i] = g2[i] * y[i];
                }
                // bias: the midpoint between the two weighted class centroids' scores
                double sPos = 0;
                double sNeg = 0;
                for (int j = 0; j < n; j++) {
                    double f = 0;
                    for (int i = 0; i < n; i++) {
                        f += coef[i] * combined[i][j];
                    }
                    if (y[j] > 0) {
                        sPos += g2[j] * f;
                    } else {
                        sNeg += g2[j] * f;
                    }
                }
                return new MultipleKernelLearning(method, kernels, scales, mu, x, null, coef, 0.5 * (sPos + sNeg), 0);
            }
            default -> throw new IllegalArgumentException(method.toString());
        }
        double[][] combined = combine(grams, mu);
        double[] muFinal = mu;
        KernelSvm svm = svmOn(combined, labels, c, tokenKernel(combined, x, kernels, scales, muFinal));
        solves[0]++;
        return new MultipleKernelLearning(method, kernels, scales, mu, x, svm, null, 0, solves[0]);
    }

    /**
     * A token {@code {NaN, i}} stands for training point {@code i}; {@code {NaN, -1 - j}} is a unit
     * probe at {@code j}.
     */
    private static double[] token(double i) {
        return new double[] {Double.NaN, i};
    }

    private static KernelFunction tokenKernel(double[][] combined, double[][] train, List<KernelFunction> kernels,
            double[] scales, double[] mu) {
        return (a, b) -> {
            boolean ta = a.length == 2 && Double.isNaN(a[0]);
            boolean tb = b.length == 2 && Double.isNaN(b[0]);
            if (ta && tb) {
                int i = (int) a[1];
                int j = (int) b[1];
                if (i < 0 || j < 0) {
                    return i < 0 ? (-1 - i == j ? 1 : 0) : (-1 - j == i ? 1 : 0);
                }
                return combined[i][j];
            }
            double[] pa = ta ? train[(int) a[1]] : a;
            double[] pb = tb ? train[(int) b[1]] : b;
            double s = 0;
            for (int k = 0; k < mu.length; k++) {
                if (mu[k] > 0) {
                    s += mu[k] * scales[k] * kernels.get(k).value(pa, pb);
                }
            }
            return s;
        };
    }

    private static KernelSvm svmOn(double[][] combined, int[] labels, double c, KernelFunction kernel) {
        int n = combined.length;
        double[][] tokens = new double[n][];
        for (int i = 0; i < n; i++) {
            tokens[i] = token(i);
        }
        return KernelSvm.fit(tokens, labels, 2, kernel, c, 1e-3);
    }

    /** Reads {@code w = alpha y} (signed for class 0) and the dual objective back from a token-trained SVM. */
    private static double[] dualCoefficients(KernelSvm svm, int n) {
        double rho = -svm.decision(token(-1 - n));
        double[] w = new double[n];
        for (int j = 0; j < n; j++) {
            w[j] = svm.decision(token(-1 - j)) + rho;
        }
        return w;
    }

    private static double[] simpleMkl(double[][][] grams, int[] labels, double[] start, double c, int iterations,
            int[] solves) {
        int m = grams.length;
        int n = labels.length;
        double[] mu = start.clone();
        double[] w = dual(grams, labels, mu, c, solves);
        double j = objective(grams, mu, w);
        double step = 0.5;
        for (int it = 0; it < iterations; it++) {
            double[] g = new double[m];
            double gmax = 0;
            for (int k = 0; k < m; k++) {
                g[k] = -0.5 * quadratic(grams[k], w, null);
                gmax = Math.max(gmax, Math.abs(g[k]));
            }
            // the reduced gradient: centred on the largest weight's coordinate
            double mean = 0;
            for (int k = 0; k < m; k++) {
                mean += g[k] / m;
            }
            boolean improved = false;
            for (int tries = 0; tries < 6; tries++) {
                double[] cand = new double[m];
                for (int k = 0; k < m; k++) {
                    cand[k] = mu[k] - step * (g[k] - mean) / Math.max(1e-12, gmax);
                }
                cand = projectSimplex(cand, 1);
                double[] wc = dual(grams, labels, cand, c, solves);
                double jc = objective(grams, cand, wc);
                if (jc < j - 1e-9 * Math.abs(j)) {
                    mu = cand;
                    w = wc;
                    j = jc;
                    step = Math.min(1, step * 1.5);
                    improved = true;
                    break;
                }
                step *= 0.5;
            }
            if (!improved || step < 1e-4) {
                break;
            }
        }
        for (int k = 0; k < m; k++) {
            mu[k] = mu[k] < 1e-6 ? 0 : mu[k];
        }
        double s = 0;
        for (double v : mu) {
            s += v;
        }
        for (int k = 0; k < m; k++) {
            mu[k] /= s;
        }
        return mu;
    }

    private static double[] dual(double[][][] grams, int[] labels, double[] mu, double c, int[] solves) {
        double[][] combined = combine(grams, mu);
        KernelSvm svm = svmOn(combined, labels, c, tokenKernel(combined, null, List.of(), new double[0], mu));
        solves[0]++;
        return dualCoefficients(svm, labels.length);
    }

    /** {@code J = sum |w| - 1/2 w^T K_mu w}, the SVM dual optimum. */
    private static double objective(double[][][] grams, double[] mu, double[] w) {
        double s = 0;
        for (double v : w) {
            s += Math.abs(v);
        }
        for (int k = 0; k < grams.length; k++) {
            if (mu[k] > 0) {
                s -= 0.5 * mu[k] * quadratic(grams[k], w, null);
            }
        }
        return s;
    }

    /** {@code v^T diag(y) K diag(y) v}, or {@code v^T K v} when {@code y} is null. */
    private static double quadratic(double[][] k, double[] v, double[] y) {
        int n = v.length;
        double s = 0;
        for (int i = 0; i < n; i++) {
            double vi = y == null ? v[i] : v[i] * y[i];
            if (vi == 0) {
                continue;
            }
            double r = 0;
            for (int j = 0; j < n; j++) {
                r += k[i][j] * (y == null ? v[j] : v[j] * y[j]);
            }
            s += vi * r;
        }
        return s;
    }

    private static double[][] combine(double[][][] grams, double[] mu) {
        int n = grams[0].length;
        double[][] out = new double[n][n];
        for (int k = 0; k < grams.length; k++) {
            if (mu[k] == 0) {
                continue;
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    out[i][j] += mu[k] * grams[k][i][j];
                }
            }
        }
        return out;
    }

    /** EasyMKL's QP over the two class simplices, by accelerated projected gradient. */
    private static double[] easyGamma(double[][] k, double[] y, double lambda) {
        int n = y.length;
        double[][] q = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                q[i][j] = (1 - lambda) * y[i] * y[j] * k[i][j];
            }
            q[i][i] += lambda;
        }
        double lip = 2 * largestEigenvalue(q);
        int pos = 0;
        for (double v : y) {
            pos += v > 0 ? 1 : 0;
        }
        double[] gamma = new double[n];
        for (int i = 0; i < n; i++) {
            gamma[i] = y[i] > 0 ? 1.0 / pos : 1.0 / (n - pos);
        }
        double[] z = gamma.clone();
        double t = 1;
        for (int it = 0; it < 400; it++) {
            double[] grad = matVec(q, z);
            double[] next = new double[n];
            for (int i = 0; i < n; i++) {
                next[i] = z[i] - 2 * grad[i] / lip;
            }
            next = projectClasses(next, y);
            double tn = 0.5 * (1 + Math.sqrt(1 + 4 * t * t));
            double change = 0;
            for (int i = 0; i < n; i++) {
                double v = next[i] + (t - 1) / tn * (next[i] - gamma[i]);
                change += Math.abs(next[i] - gamma[i]);
                z[i] = v;
            }
            gamma = next;
            t = tn;
            if (change < 1e-9) {
                break;
            }
        }
        return gamma;
    }

    private static double[] projectClasses(double[] v, double[] y) {
        int n = v.length;
        int pos = 0;
        for (double s : y) {
            pos += s > 0 ? 1 : 0;
        }
        double[] a = new double[pos];
        double[] b = new double[n - pos];
        int ia = 0;
        int ib = 0;
        for (int i = 0; i < n; i++) {
            if (y[i] > 0) {
                a[ia++] = v[i];
            } else {
                b[ib++] = v[i];
            }
        }
        a = projectSimplex(a, 1);
        b = projectSimplex(b, 1);
        double[] out = new double[n];
        ia = 0;
        ib = 0;
        for (int i = 0; i < n; i++) {
            out[i] = y[i] > 0 ? a[ia++] : b[ib++];
        }
        return out;
    }

    /** ALIGNF: non-negative least squares {@code min_v>=0 v^T M v - 2 v^T a}, normalised to the simplex. */
    private static double[] alignF(double[][][] grams, double[] y) {
        int m = grams.length;
        int n = y.length;
        double[][][] centred = new double[m][][];
        for (int k = 0; k < m; k++) {
            centred[k] = centre(grams[k]);
        }
        double[][] mm = new double[m][m];
        double[] a = new double[m];
        for (int k = 0; k < m; k++) {
            for (int l = k; l < m; l++) {
                double s = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        s += centred[k][i][j] * centred[l][i][j];
                    }
                }
                mm[k][l] = s;
                mm[l][k] = s;
            }
            a[k] = quadratic(centred[k], y, null);
        }
        double[] v = nonNegativeLeastSquares(mm, a, 2000);
        double s = 0;
        for (double e : v) {
            s += e;
        }
        if (s <= 0) {
            double[] u = new double[m];
            java.util.Arrays.fill(u, 1.0 / m);
            return u;
        }
        for (int k = 0; k < m; k++) {
            v[k] /= s;
        }
        return v;
    }

    /**
     * Minimises {@code v^T M v - 2 v^T a} over {@code v >= 0} by accelerated projected gradient.
     *
     * @param mm a symmetric positive semi-definite matrix
     * @param a the linear term
     * @param iterations iterations
     * @return the minimiser
     */
    static double[] nonNegativeLeastSquares(double[][] mm, double[] a, int iterations) {
        int m = a.length;
        double lip = 2 * largestEigenvalue(mm) + 1e-12;
        double[] v = new double[m];
        double[] z = new double[m];
        double t = 1;
        for (int it = 0; it < iterations; it++) {
            double[] mz = matVec(mm, z);
            double[] next = new double[m];
            for (int k = 0; k < m; k++) {
                next[k] = Math.max(0, z[k] - 2 * (mz[k] - a[k]) / lip);
            }
            double tn = 0.5 * (1 + Math.sqrt(1 + 4 * t * t));
            for (int k = 0; k < m; k++) {
                z[k] = next[k] + (t - 1) / tn * (next[k] - v[k]);
            }
            v = next;
            t = tn;
        }
        return v;
    }

    private static double[][] centre(double[][] k) {
        int n = k.length;
        double[] row = new double[n];
        double all = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                row[i] += k[i][j] / n;
            }
            all += row[i] / n;
        }
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                out[i][j] = k[i][j] - row[i] - row[j] + all;
            }
        }
        return out;
    }

    private static double largestEigenvalue(double[][] a) {
        int n = a.length;
        double[] v = new double[n];
        java.util.Arrays.fill(v, 1 / Math.sqrt(n));
        double lam = 0;
        for (int it = 0; it < 50; it++) {
            double[] w = matVec(a, v);
            double norm = 0;
            for (double e : w) {
                norm += e * e;
            }
            norm = Math.sqrt(norm);
            if (norm == 0) {
                return 0;
            }
            for (int i = 0; i < n; i++) {
                v[i] = w[i] / norm;
            }
            lam = norm;
        }
        return lam * 1.05;
    }

    private static double[] matVec(double[][] a, double[] v) {
        double[] out = new double[v.length];
        for (int i = 0; i < a.length; i++) {
            double s = 0;
            for (int j = 0; j < v.length; j++) {
                s += a[i][j] * v[j];
            }
            out[i] = s;
        }
        return out;
    }

    /**
     * The Euclidean projection onto {@code {v >= 0, sum v = total}} (Duchi et al. 2008).
     *
     * @param v a vector
     * @param total the simplex size
     * @return the projection
     */
    static double[] projectSimplex(double[] v, double total) {
        int n = v.length;
        double[] s = v.clone();
        java.util.Arrays.sort(s);
        double cum = 0;
        double theta = 0;
        for (int i = n - 1; i >= 0; i--) {
            cum += s[i];
            double t = (cum - total) / (n - i);
            if (i == 0 || s[i - 1] <= t) {
                theta = t;
                break;
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = Math.max(0, v[i] - theta);
        }
        return out;
    }

    /**
     * The decision value, positive for class 1.
     *
     * @param p a point
     * @return the score
     */
    public double decision(double[] p) {
        if (svm != null) {
            return -svm.decision(p);
        }
        double s = -bias;
        for (int i = 0; i < x.length; i++) {
            if (coefficients[i] == 0) {
                continue;
            }
            double k = 0;
            for (int m = 0; m < mu.length; m++) {
                if (mu[m] > 0) {
                    k += mu[m] * scales[m] * kernels.get(m).value(x[i], p);
                }
            }
            s += coefficients[i] * k;
        }
        return s;
    }

    /**
     * Predicts labels.
     *
     * @param points points
     * @return {@code 1} where the decision is positive, else {@code 0}
     */
    public int[] predict(double[][] points) {
        int[] out = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            out[i] = decision(points[i]) > 0 ? 1 : 0;
        }
        return out;
    }

    /** @return the learned kernel weights on the simplex (base kernels scaled to unit mean diagonal) */
    public double[] weights() {
        return mu.clone();
    }

    /** @return the method */
    public Method method() {
        return method;
    }

    /** @return the SVM solves the fit used (0 for EasyMKL) */
    public int svmSolves() {
        return svmSolves;
    }
}
