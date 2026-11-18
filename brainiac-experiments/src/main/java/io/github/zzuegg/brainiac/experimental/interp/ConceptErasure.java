package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.experimental.nn.LinearProbe;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Linear concept erasure: remove a concept (a protected attribute, a spurious feature) from
 * representations so no linear classifier can recover it, changing them as little as
 * possible. Every eraser here is affine, {@code x' = x - M (x - mu)}.
 *
 * <ul>
 *   <li>{@link #leace} (LEAst-squares Concept Erasure, Belrose et al., NeurIPS 2023) is the
 *       closed-form optimum: with the whitening {@code W = Sigma_xx^(-1/2)} and the
 *       orthogonal projection {@code P} onto the column space of {@code W Sigma_xz} (the
 *       cross-covariance with the one-hot concept), {@code M = W^+ P W}. Afterwards the
 *       class-conditional means coincide, which provably stops every linear probe from
 *       beating a constant predictor, with the least mean-squared change.</li>
 *   <li>{@link #inlp} (Iterative Nullspace Projection, Ravfogel et al., ACL 2020)
 *       repeatedly trains a {@link LinearProbe} and projects the data onto the nullspace of
 *       its weights; the probe's weight differences are read back exactly from its
 *       probabilities (the log-odds are affine), so the probe class is used unchanged.</li>
 *   <li>{@link #meanDifference} projects out the normalised difference of the two concept
 *       means (the {@code PersonaVector} ablation).</li>
 * </ul>
 *
 * <p>An "oracle-free" LEACE is {@link #leace} fitted on a labelled subset only.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2306.03819">arXiv:2306.03819</a>,
 * <a href="https://arxiv.org/abs/2004.07667">arXiv:2004.07667</a>.
 */
public final class ConceptErasure {

    private ConceptErasure() {
    }

    /**
     * An affine eraser {@code x' = x - M (x - mu)}.
     *
     * @param mean {@code mu}
     * @param correction {@code M}, {@code [d][d]}
     * @param name what fitted it
     */
    public record Eraser(double[] mean, double[][] correction, String name) {

        /**
         * Erases one vector.
         *
         * @param x the representation
         * @return the erased representation
         */
        public float[] apply(float[] x) {
            int d = mean.length;
            double[] c = new double[d];
            for (int j = 0; j < d; j++) {
                c[j] = x[j] - mean[j];
            }
            float[] out = new float[d];
            for (int i = 0; i < d; i++) {
                double s = 0;
                for (int j = 0; j < d; j++) {
                    s += correction[i][j] * c[j];
                }
                out[i] = (float) (x[i] - s);
            }
            return out;
        }

        /**
         * Erases many vectors.
         *
         * @param xs the representations
         * @return the erased representations
         */
        public float[][] apply(float[][] xs) {
            float[][] out = new float[xs.length][];
            for (int i = 0; i < xs.length; i++) {
                out[i] = apply(xs[i]);
            }
            return out;
        }

        /**
         * The mean squared change the eraser makes on data.
         *
         * @param xs the representations
         * @return the mean of {@code |x' - x|^2}
         */
        public double meanSquaredChange(float[][] xs) {
            double s = 0;
            for (float[] x : xs) {
                float[] e = apply(x);
                for (int j = 0; j < x.length; j++) {
                    s += (e[j] - x[j]) * (e[j] - x[j]);
                }
            }
            return s / xs.length;
        }
    }

    /**
     * The identity, for "no erasure" rows.
     *
     * @param d the width
     * @return the eraser that changes nothing
     */
    public static Eraser none(int d) {
        return new Eraser(new double[d], new double[d][d], "none");
    }

    /**
     * Fits LEACE.
     *
     * @param x representations {@code [n][d]}
     * @param z concept labels {@code 0..classes-1}
     * @param classes the number of concept values
     * @return the eraser
     */
    public static Eraser leace(float[][] x, int[] z, int classes) {
        double[][] xd = InterpMath.toDouble(x);
        int n = xd.length;
        double[] mu = InterpMath.mean(xd);
        double[][] xc = InterpMath.centre(xd);
        double[][] zc = new double[n][classes];
        double[] zm = new double[classes];
        for (int i = 0; i < n; i++) {
            zm[z[i]] += 1.0 / n;
        }
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < classes; c++) {
                zc[i][c] = (z[i] == c ? 1 : 0) - zm[c];
            }
        }
        double[][] sxx = InterpMath.crossProduct(xc, xc, n);
        double[][] sxz = InterpMath.crossProduct(xc, zc, n);
        double[][] w = InterpMath.symmetricPower(sxx, -0.5, 1e-9);
        double[][] wInv = InterpMath.symmetricPower(sxx, 0.5, 1e-9);
        double[][] u = InterpMath.columnBasis(InterpMath.matmul(w, sxz), 1e-6);
        double[][] p = InterpMath.matmul(u, InterpMath.transpose(u));
        double[][] m = InterpMath.matmul(InterpMath.matmul(wInv, p), w);
        return new Eraser(mu, m, "LEACE");
    }

    /**
     * Fits INLP.
     *
     * @param x representations {@code [n][d]}
     * @param z concept labels
     * @param classes the number of concept values
     * @param iterations probes to train and project out
     * @param epochs full-batch Adam steps per probe
     * @return the eraser
     */
    public static Eraser inlp(float[][] x, int[] z, int classes, int iterations, int epochs) {
        double[][] xd = InterpMath.toDouble(x);
        int d = xd[0].length;
        double[] mu = InterpMath.mean(xd);
        double[][] keep = InterpMath.identity(d);
        List<double[]> removed = new ArrayList<>();
        float[][] current = x;
        for (int it = 0; it < iterations; it++) {
            LinearProbe probe = LinearProbe.fit(current, z, classes, epochs, 0.05, 1e-4);
            for (double[] dir : probeDirections(probe, d, classes)) {
                double[] v = dir.clone();
                for (double[] r : removed) {
                    double c = InterpMath.dot(v, r);
                    for (int j = 0; j < d; j++) {
                        v[j] -= c * r[j];
                    }
                }
                double norm = Math.sqrt(InterpMath.dot(v, v));
                if (norm > 1e-8) {
                    for (int j = 0; j < d; j++) {
                        v[j] /= norm;
                    }
                    removed.add(v);
                }
            }
            keep = InterpMath.identity(d);
            for (double[] r : removed) {
                for (int i = 0; i < d; i++) {
                    for (int j = 0; j < d; j++) {
                        keep[i][j] -= r[i] * r[j];
                    }
                }
            }
            double[][] m = correctionOf(keep);
            current = new Eraser(mu, m, "INLP").apply(x);
        }
        return new Eraser(mu, correctionOf(keep), "INLP x" + iterations);
    }

    private static double[][] correctionOf(double[][] keep) {
        int d = keep.length;
        double[][] m = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                m[i][j] = (i == j ? 1 : 0) - keep[i][j];
            }
        }
        return m;
    }

    /**
     * The weight differences {@code w_c - w_0} of a softmax probe, read from its
     * probabilities: {@code log p_c(x) - log p_0(x)} is affine in {@code x}.
     *
     * @param probe the probe
     * @param d its input width
     * @param classes its classes
     * @return {@code classes - 1} weight-difference directions of length {@code d}
     */
    public static List<double[]> probeDirections(LinearProbe probe, int d, int classes) {
        float[] zero = new float[d];
        double[] p0 = probe.probabilities(zero);
        List<double[]> dirs = new ArrayList<>();
        double[][] w = new double[classes][d];
        for (int j = 0; j < d; j++) {
            float[] e = new float[d];
            e[j] = 1;
            double[] p = probe.probabilities(e);
            for (int c = 1; c < classes; c++) {
                w[c][j] = Math.log(Math.max(p[c], 1e-300) / Math.max(p[0], 1e-300))
                        - Math.log(Math.max(p0[c], 1e-300) / Math.max(p0[0], 1e-300));
            }
        }
        for (int c = 1; c < classes; c++) {
            dirs.add(w[c]);
        }
        return dirs;
    }

    /**
     * Projects out the normalised difference of the two concept means, as
     * {@code PersonaVector.ablate} does at a layer.
     *
     * @param x representations
     * @param z binary concept labels
     * @return the eraser
     */
    public static Eraser meanDifference(float[][] x, int[] z) {
        int d = x[0].length;
        double[] a = new double[d];
        double[] b = new double[d];
        int na = 0;
        int nb = 0;
        for (int i = 0; i < x.length; i++) {
            double[] t = z[i] == 1 ? a : b;
            for (int j = 0; j < d; j++) {
                t[j] += x[i][j];
            }
            if (z[i] == 1) {
                na++;
            } else {
                nb++;
            }
        }
        double[] u = new double[d];
        double norm = 0;
        for (int j = 0; j < d; j++) {
            u[j] = a[j] / Math.max(1, na) - b[j] / Math.max(1, nb);
            norm += u[j] * u[j];
        }
        norm = Math.sqrt(Math.max(norm, 1e-300));
        double[][] m = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                m[i][j] = u[i] / norm * u[j] / norm;
            }
        }
        return new Eraser(new double[d], m, "mean difference");
    }

    /**
     * A random subset of labelled rows, for fitting an oracle-free eraser on few labels.
     *
     * @param n the number of rows
     * @param k the subset size
     * @param seed the seed
     * @return row indices
     */
    public static int[] subset(int n, int k, long seed) {
        return new SplittableRandom(seed).ints(0, n).distinct().limit(Math.min(n, k)).toArray();
    }
}
