package io.github.zzuegg.brainiac.experimental.tsc;

import java.util.SplittableRandom;

/**
 * k-Shape (Paparrizos and Gravano, SIGMOD 2015): k-means for series under the shape-based distance
 * {@code SBD(x, y) = 1 - max_w NCC_w(x, y)}, one minus the largest coefficient-normalised
 * cross-correlation over all shifts, which makes the distance invariant to scaling (by
 * z-normalisation) and to shifts. The centroid of a cluster is the shape that maximises the
 * summed squared correlation with its members after aligning them to the previous centroid: the
 * leading eigenvector of {@code Q^T S Q}, with {@code S} the members' scatter matrix and
 * {@code Q = I - 1 1^T / m} the centring matrix, found by power iteration (the matrix is positive
 * semi-definite). Cross-correlations are computed directly in {@code O(m^2)} per pair, which is
 * cheaper than an FFT at the lengths used here.
 */
public final class KShape {

    /**
     * A clustering.
     *
     * @param labels cluster per series
     * @param centroids the z-normalised centroids
     */
    public record Clustering(int[] labels, double[][] centroids) {
    }

    private KShape() {
    }

    /**
     * Z-normalises a series.
     *
     * @param x the series
     * @return {@code (x - mean) / sd}
     */
    public static double[] zNormalise(double[] x) {
        double mean = 0;
        for (double v : x) {
            mean += v / x.length;
        }
        double var = 0;
        for (double v : x) {
            var += (v - mean) * (v - mean) / x.length;
        }
        double sd = Math.sqrt(var);
        double[] out = new double[x.length];
        for (int t = 0; t < x.length; t++) {
            out[t] = sd > 1e-12 ? (x[t] - mean) / sd : 0;
        }
        return out;
    }

    /**
     * The shape-based distance and the copy of {@code y} shifted to best match {@code x}.
     *
     * @param x the reference
     * @param y the series to align, the same length
     * @return {@code {distance}} followed by the aligned {@code y} (zero padded)
     */
    public static double[] sbd(double[] x, double[] y) {
        int m = x.length;
        double nx = 0;
        double ny = 0;
        for (int t = 0; t < m; t++) {
            nx += x[t] * x[t];
            ny += y[t] * y[t];
        }
        double norm = Math.sqrt(nx * ny);
        double best = Double.NEGATIVE_INFINITY;
        int shift = 0;
        for (int s = -(m - 1); s < m; s++) {
            double c = 0;
            for (int t = Math.max(0, s); t < Math.min(m, m + s); t++) {
                c += x[t] * y[t - s];
            }
            if (c > best) {
                best = c;
                shift = s;
            }
        }
        double[] out = new double[m + 1];
        out[0] = norm > 0 ? 1 - best / norm : 1;
        for (int t = 0; t < m; t++) {
            int src = t - shift;
            out[t + 1] = src >= 0 && src < m ? y[src] : 0;
        }
        return out;
    }

    /**
     * Clusters series.
     *
     * @param series the series (one length)
     * @param k clusters
     * @param iterations the most refinement iterations
     * @param seed the seed of the random initial assignment
     * @return the clustering
     */
    public static Clustering fit(double[][] series, int k, int iterations, long seed) {
        int n = series.length;
        int m = series[0].length;
        double[][] x = new double[n][];
        for (int i = 0; i < n; i++) {
            x[i] = zNormalise(series[i]);
        }
        SplittableRandom r = new SplittableRandom(seed);
        int[] label = new int[n];
        for (int i = 0; i < n; i++) {
            label[i] = r.nextInt(k);
        }
        double[][] centroids = new double[k][m];
        for (int it = 0; it < iterations; it++) {
            for (int c = 0; c < k; c++) {
                centroids[c] = centroid(x, label, c, centroids[c]);
            }
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY;
                int arg = label[i];
                for (int c = 0; c < k; c++) {
                    double d = sbd(centroids[c], x[i])[0];
                    if (d < best) {
                        best = d;
                        arg = c;
                    }
                }
                changed |= arg != label[i];
                label[i] = arg;
            }
            if (!changed) {
                break;
            }
        }
        return new Clustering(label, centroids);
    }

    /** Shape extraction: the leading eigenvector of the centred scatter of aligned members. */
    static double[] centroid(double[][] x, int[] label, int c, double[] previous) {
        int m = x[0].length;
        boolean zero = true;
        for (double v : previous) {
            zero &= v == 0;
        }
        double[][] s = new double[m][m];
        int count = 0;
        for (int i = 0; i < x.length; i++) {
            if (label[i] != c) {
                continue;
            }
            double[] a = x[i];
            if (!zero) {
                double[] al = sbd(previous, x[i]);
                a = java.util.Arrays.copyOfRange(al, 1, m + 1);
            }
            a = zNormalise(a);
            for (int p = 0; p < m; p++) {
                for (int q = 0; q < m; q++) {
                    s[p][q] += a[p] * a[q];
                }
            }
            count++;
        }
        if (count == 0) {
            return previous;
        }
        // M = Q^T S Q with Q = I - 1/m: centre rows and columns of S
        double[] rowMean = new double[m];
        double all = 0;
        for (int p = 0; p < m; p++) {
            for (int q = 0; q < m; q++) {
                rowMean[p] += s[p][q] / m;
            }
            all += rowMean[p] / m;
        }
        double[][] mm = new double[m][m];
        for (int p = 0; p < m; p++) {
            for (int q = 0; q < m; q++) {
                mm[p][q] = s[p][q] - rowMean[p] - rowMean[q] + all;
            }
        }
        double[] v = leadingEigenvector(mm, zero ? null : previous);
        // pick the sign closer to the members
        double plus = 0;
        double minus = 0;
        for (int i = 0; i < x.length; i++) {
            if (label[i] == c) {
                double[] neg = new double[m];
                for (int p = 0; p < m; p++) {
                    neg[p] = -v[p];
                }
                plus += ElasticDistances.euclidean(x[i], zNormalise(v));
                minus += ElasticDistances.euclidean(x[i], zNormalise(neg));
            }
        }
        if (minus < plus) {
            for (int p = 0; p < m; p++) {
                v[p] = -v[p];
            }
        }
        return zNormalise(v);
    }

    /** Power iteration for the leading eigenvector of a positive semi-definite matrix. */
    private static double[] leadingEigenvector(double[][] a, double[] start) {
        int m = a.length;
        double[] v = new double[m];
        for (int p = 0; p < m; p++) {
            v[p] = start != null ? start[p] + 1e-3 * Math.sin(p + 1.0) : Math.sin(0.7 * p + 1.0);
        }
        for (int it = 0; it < 300; it++) {
            double[] w = new double[m];
            for (int p = 0; p < m; p++) {
                double s = 0;
                for (int q = 0; q < m; q++) {
                    s += a[p][q] * v[q];
                }
                w[p] = s;
            }
            double norm = 0;
            for (double x : w) {
                norm += x * x;
            }
            norm = Math.sqrt(norm);
            if (norm == 0) {
                return v;
            }
            double change = 0;
            for (int p = 0; p < m; p++) {
                w[p] /= norm;
                change += (w[p] - v[p]) * (w[p] - v[p]);
            }
            v = w;
            if (change < 1e-12) {
                break;
            }
        }
        return v;
    }
}
