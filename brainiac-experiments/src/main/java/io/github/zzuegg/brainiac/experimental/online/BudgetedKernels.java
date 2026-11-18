package io.github.zzuegg.brainiac.experimental.online;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;
import io.github.zzuegg.brainiac.experimental.stream.OnlineClassifier;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Online kernel classification with a bounded number of support vectors, multi-class with one
 * coefficient per class on every support vector and an RBF kernel {@code exp(-γ ||x - x'||^2)}.
 * Plain kernel methods add a support vector per mistake and grow without bound; the budgeted
 * variants differ in what they forget.
 *
 * <ul>
 *   <li>{@link Kind#KERNEL_PERCEPTRON}: on a mistake add {@code +1} for the true and {@code -1}
 *       for the predicted class; unbounded (the budget is ignored).</li>
 *   <li>{@link Kind#NORMA}: kernelised SGD on the multi-class hinge loss with shrinkage
 *       {@code α ← (1 - ηλ) α} every step, truncating the oldest support vector beyond the
 *       budget (Kivinen, Smola and Williamson, IEEE TSP 2004).</li>
 *   <li>{@link Kind#FORGETRON}: perceptron updates; when over budget, shrink all coefficients
 *       by {@code φ} and remove the oldest (Dekel, Shalev-Shwartz and Singer, SIAM J. Comput.
 *       2008; constant shrink instead of the paper's adaptive one).</li>
 *   <li>{@link Kind#BSGD_REMOVE} and {@link Kind#BSGD_MERGE}: Pegasos-style budgeted SGD (Wang,
 *       Crammer and Vucetic, JMLR 2012) that removes the support vector of smallest norm, or
 *       merges it with its nearest neighbour into {@code z = h x_m + (1 - h) x_n} with {@code h}
 *       found by golden-section search and coefficients {@code α_m k(x_m, z) + α_n k(x_n, z)}.</li>
 *   <li>{@link Kind#ONLINE_RFF}: random Fourier features {@code sqrt(2/D) cos(ω·x + b)}, with
 *       {@code ω ~ N(0, 2γ)}, and a linear multi-class hinge model on them (Lu et al., JMLR
 *       2016); fixed memory, no support vectors.</li>
 * </ul>
 */
public final class BudgetedKernels implements OnlineClassifier {

    /** The update and budget rule. */
    public enum Kind {
        /** unbounded kernel perceptron */
        KERNEL_PERCEPTRON,
        /** NORMA: shrinkage and truncation */
        NORMA,
        /** the Forgetron: shrink and forget the oldest */
        FORGETRON,
        /** budgeted SGD, removing the smallest support vector */
        BSGD_REMOVE,
        /** budgeted SGD, merging the smallest support vector into its neighbour */
        BSGD_MERGE,
        /** online random Fourier features */
        ONLINE_RFF
    }

    private final Kind kind;
    private final int classes;
    private final int budget;
    private final double gamma;
    private final double lambda;
    private final double rate;
    private final List<double[]> points = new ArrayList<>();
    private final List<double[]> alphas = new ArrayList<>();
    private final double[][] omega;
    private final double[] phase;
    private final double[][] linear;
    private int t;

    private BudgetedKernels(Kind kind, int features, int classes, int budget, double gamma, double lambda, double rate,
            long seed) {
        this.kind = kind;
        this.classes = classes;
        this.budget = budget;
        this.gamma = gamma;
        this.lambda = lambda;
        this.rate = rate;
        if (kind == Kind.ONLINE_RFF) {
            SplittableRandom r = new SplittableRandom(seed);
            omega = new double[budget][features];
            phase = new double[budget];
            double sd = Math.sqrt(2 * gamma);
            for (int k = 0; k < budget; k++) {
                for (int j = 0; j < features; j++) {
                    omega[k][j] = r.nextGaussian() * sd;
                }
                phase[k] = r.nextDouble() * 2 * Math.PI;
            }
            linear = new double[classes][budget + 1];
        } else {
            omega = null;
            phase = null;
            linear = null;
        }
    }

    /**
     * A learner.
     *
     * @param kind the rule
     * @param features the input dimension
     * @param classes the number of classes
     * @param budget the support vectors kept (for {@link Kind#ONLINE_RFF}, the number of features)
     * @param gamma the RBF width
     * @param lambda the regulariser of NORMA and BSGD
     * @param rate the step size of NORMA, BSGD and online RFF (BSGD uses {@code 1 / (λ t)} if 0)
     * @param seed the seed of the random features
     * @return the learner
     */
    public static BudgetedKernels of(Kind kind, int features, int classes, int budget, double gamma, double lambda,
            double rate, long seed) {
        return new BudgetedKernels(kind, features, classes, budget, gamma, lambda, rate, seed);
    }

    private double kernel(double[] a, double[] b) {
        double d = 0;
        for (int j = 0; j < a.length; j++) {
            double v = a[j] - b[j];
            d += v * v;
        }
        return Math.exp(-gamma * d);
    }

    private double[] features(double[] x) {
        double[] f = new double[omega.length + 1];
        double s = Math.sqrt(2.0 / omega.length);
        for (int k = 0; k < omega.length; k++) {
            double dot = phase[k];
            for (int j = 0; j < x.length; j++) {
                dot += omega[k][j] * x[j];
            }
            f[k] = s * Math.cos(dot);
        }
        f[omega.length] = 1;
        return f;
    }

    /**
     * The class scores.
     *
     * @param x an input
     * @return one score per class
     */
    public double[] scores(double[] x) {
        double[] s = new double[classes];
        if (kind == Kind.ONLINE_RFF) {
            double[] f = features(x);
            for (int c = 0; c < classes; c++) {
                for (int k = 0; k < f.length; k++) {
                    s[c] += linear[c][k] * f[k];
                }
            }
            return s;
        }
        for (int i = 0; i < points.size(); i++) {
            double k = kernel(points.get(i), x);
            double[] a = alphas.get(i);
            for (int c = 0; c < classes; c++) {
                s[c] += a[c] * k;
            }
        }
        return s;
    }

    @Override
    public double[] predict(DriftingStreams.Sample sample) {
        double[] s = scores(sample.x());
        double max = Double.NEGATIVE_INFINITY;
        for (double v : s) {
            max = Math.max(max, v);
        }
        double[] p = new double[classes];
        double z = 0;
        for (int c = 0; c < classes; c++) {
            p[c] = Math.exp(Math.max(-50, 4 * (s[c] - max)));
            z += p[c];
        }
        for (int c = 0; c < classes; c++) {
            p[c] /= z;
        }
        return p;
    }

    @Override
    public void learn(DriftingStreams.Sample sample) {
        learn(sample.x(), sample.label());
    }

    /**
     * Learns from one labelled example.
     *
     * @param x the input
     * @param y the class
     */
    public void learn(double[] x, int y) {
        t++;
        double[] s = scores(x);
        int rival = -1;
        for (int c = 0; c < classes; c++) {
            if (c != y && (rival < 0 || s[c] > s[rival])) {
                rival = c;
            }
        }
        double margin = s[y] - s[rival];
        switch (kind) {
            case KERNEL_PERCEPTRON, FORGETRON -> {
                if (margin <= 0) {
                    add(x, y, rival, 1);
                    if (kind == Kind.FORGETRON && points.size() > budget) {
                        scaleAll(0.9);
                        removeAt(0);
                    }
                }
            }
            case NORMA -> {
                scaleAll(1 - rate * lambda);
                if (margin < 1) {
                    add(x, y, rival, rate);
                }
                while (points.size() > budget) {
                    removeAt(0);
                }
            }
            case BSGD_REMOVE, BSGD_MERGE -> {
                double eta = rate > 0 ? rate : 1.0 / (lambda * t);
                scaleAll(1 - eta * lambda);
                if (margin < 1) {
                    add(x, y, rival, eta);
                }
                while (points.size() > budget) {
                    int m = smallest();
                    if (kind == Kind.BSGD_REMOVE) {
                        removeAt(m);
                    } else {
                        merge(m);
                    }
                }
            }
            case ONLINE_RFF -> {
                if (margin < 1) {
                    double[] f = features(x);
                    for (int k = 0; k < f.length; k++) {
                        linear[y][k] += rate * f[k];
                        linear[rival][k] -= rate * f[k];
                    }
                }
                for (double[] row : linear) {
                    for (int k = 0; k < row.length; k++) {
                        row[k] *= 1 - rate * lambda;
                    }
                }
            }
            default -> throw new IllegalStateException();
        }
    }

    private void add(double[] x, int y, int rival, double step) {
        double[] a = new double[classes];
        a[y] = step;
        a[rival] = -step;
        points.add(x.clone());
        alphas.add(a);
    }

    private void scaleAll(double f) {
        for (double[] a : alphas) {
            for (int c = 0; c < a.length; c++) {
                a[c] *= f;
            }
        }
    }

    private void removeAt(int i) {
        points.remove(i);
        alphas.remove(i);
    }

    private int smallest() {
        int best = 0;
        double bestNorm = Double.POSITIVE_INFINITY;
        for (int i = 0; i < alphas.size(); i++) {
            double n = 0;
            for (double v : alphas.get(i)) {
                n += v * v;
            }
            if (n < bestNorm) {
                bestNorm = n;
                best = i;
            }
        }
        return best;
    }

    /** Merges support vector m into its nearest neighbour. */
    private void merge(int m) {
        double[] xm = points.get(m);
        double[] am = alphas.get(m);
        int nIdx = -1;
        double bestK = -1;
        for (int i = 0; i < points.size(); i++) {
            if (i != m) {
                double k = kernel(xm, points.get(i));
                if (k > bestK) {
                    bestK = k;
                    nIdx = i;
                }
            }
        }
        double[] xn = points.get(nIdx);
        double[] an = alphas.get(nIdx);
        // weight of the class-wise combination: maximise ||α_m k(x_m, z) + α_n k(x_n, z)||
        double kmn = bestK;
        double lo = 0;
        double hi = 1;
        double g = (Math.sqrt(5) - 1) / 2;
        for (int it = 0; it < 30; it++) {
            double h1 = hi - g * (hi - lo);
            double h2 = lo + g * (hi - lo);
            if (mergeObjective(am, an, kmn, h1) > mergeObjective(am, an, kmn, h2)) {
                hi = h2;
            } else {
                lo = h1;
            }
        }
        double h = (lo + hi) / 2;
        double[] z = new double[xm.length];
        for (int j = 0; j < z.length; j++) {
            z[j] = h * xm[j] + (1 - h) * xn[j];
        }
        double km = kernel(xm, z);
        double kn = kernel(xn, z);
        double[] az = new double[classes];
        for (int c = 0; c < classes; c++) {
            az[c] = am[c] * km + an[c] * kn;
        }
        int hiIdx = Math.max(m, nIdx);
        int loIdx = Math.min(m, nIdx);
        removeAt(hiIdx);
        removeAt(loIdx);
        points.add(z);
        alphas.add(az);
    }

    /** {@code ||α_m k_mn^((1-h)^2) + α_n k_mn^(h^2)||^2}, since {@code k(x_m, z) = k_mn^((1-h)^2)}. */
    private static double mergeObjective(double[] am, double[] an, double kmn, double h) {
        double a = Math.pow(kmn, (1 - h) * (1 - h));
        double b = Math.pow(kmn, h * h);
        double s = 0;
        for (int c = 0; c < am.length; c++) {
            double v = am[c] * a + an[c] * b;
            s += v * v;
        }
        return s;
    }

    /** @return the number of support vectors (0 for random features) */
    public int supportVectors() {
        return points.size();
    }

    @Override
    public long bytes() {
        if (kind == Kind.ONLINE_RFF) {
            return 8L * (omega.length * (omega[0].length + 1L) + (long) classes * (omega.length + 1));
        }
        return points.isEmpty() ? 0 : 8L * points.size() * (points.get(0).length + classes);
    }
}
