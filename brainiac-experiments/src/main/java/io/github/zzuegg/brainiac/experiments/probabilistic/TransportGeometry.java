package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.data.SyntheticGraphs;
import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;
import io.github.zzuegg.brainiac.experimental.ml.GromovWasserstein;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.WassersteinBarycenter;
import io.github.zzuegg.brainiac.experimental.nn.MinibatchOptimalTransport;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Optimal-transport geometry across domains.
 *
 * <ul>
 *   <li>{@link #graphMatching}: a {@code SyntheticGraphs} community graph against a randomly
 *       permuted copy with a fraction of its edges rewired. Gromov-Wasserstein on shortest-path
 *       distances against Umeyama spectral matching (absolute adjacency eigenvectors and a
 *       Hungarian assignment). Score: node-matching accuracy.</li>
 *   <li>{@link #embeddingAlignment}: a clustered, anisotropic 8-D embedding against a rotated,
 *       noisy, permuted copy (the setting of aligning embedding spaces trained from different
 *       seeds). GW, which needs no anchors, against orthogonal Procrustes fitted on k known
 *       anchor pairs. Score: precision@1 of the recovered correspondence.</li>
 *   <li>{@link #barycenter}: the average of {@code ShapeImages} glyphs of one class at random
 *       positions. Fixed-support Wasserstein barycenter against the Euclidean (pixel) mean.
 *       Scores: mean transport cost (pixels^2) to the inputs and entropy of the result.</li>
 * </ul>
 */
public final class TransportGeometry {

    private TransportGeometry() {
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private static double accuracy(int[] found, int[] truth) {
        int ok = 0;
        for (int i = 0; i < truth.length; i++) {
            ok += found[i] == truth[i] ? 1 : 0;
        }
        return (double) ok / truth.length;
    }

    private static int[] spectralMatching(float[] a1, float[] a2, int n) {
        double[][] m1 = new double[n][n];
        double[][] m2 = new double[n][n];
        for (int i = 0; i < n * n; i++) {
            m1[i / n][i % n] = a1[i];
            m2[i / n][i % n] = a2[i];
        }
        MlMatrices.Eigen e1 = MlMatrices.eigh(m1);
        MlMatrices.Eigen e2 = MlMatrices.eigh(m2);
        double[][] cost = new double[n][n];
        for (int k = 0; k < n; k++) {
            double[] u = e1.vector(k);
            double[] v = e2.vector(k);
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    cost[i][j] -= Math.abs(u[i]) * Math.abs(v[j]);
                }
            }
        }
        return MinibatchOptimalTransport.assignCost(cost);
    }

    /**
     * Graph matching under edge noise.
     *
     * @param noises the fractions of edges rewired
     * @param seed the seed
     * @return {@code "GW"}/{@code "spectral"} to accuracy per noise level
     */
    public static Map<String, double[]> graphMatching(double[] noises, long seed) {
        SyntheticGraphs.Graph g = SyntheticGraphs.communities(4, 12, 0.5, 0.06, 0, seed);
        int n = g.nodes();
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[] gw = new double[noises.length];
        double[] spectral = new double[noises.length];
        for (int q = 0; q < noises.length; q++) {
            float[] noisy = g.adjacency().clone();
            int edges = g.edges();
            int rewire = (int) Math.round(noises[q] * edges);
            for (int e = 0; e < rewire; e++) {
                int i;
                int j;
                do {
                    i = r.nextInt(n);
                    j = r.nextInt(n);
                } while (i == j || noisy[i * n + j] == 0);
                noisy[i * n + j] = 0;
                noisy[j * n + i] = 0;
                do {
                    i = r.nextInt(n);
                    j = r.nextInt(n);
                } while (i == j || noisy[i * n + j] != 0);
                noisy[i * n + j] = 1;
                noisy[j * n + i] = 1;
            }
            int[] p = permutation(n, r);
            float[] permuted = new float[n * n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    permuted[p[i] * n + p[j]] = noisy[i * n + j];
                }
            }
            double[][] c1 = GromovWasserstein.shortestPaths(g.adjacency(), n);
            double[][] c2 = GromovWasserstein.shortestPaths(permuted, n);
            gw[q] = accuracy(GromovWasserstein.align(c1, c2, 0.002, 40, 3, seed).matching(), p);
            spectral[q] = accuracy(spectralMatching(g.adjacency(), permuted, n), p);
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("Gromov-Wasserstein", gw);
        out.put("spectral (Umeyama)", spectral);
        return out;
    }

    private static double[][] randomRotation(int d, SplittableRandom r) {
        double[][] q = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int k = 0; k < d; k++) {
                q[i][k] = r.nextGaussian();
            }
            for (int j = 0; j < i; j++) {
                double dot = MlMatrices.dot(q[i], q[j]);
                for (int k = 0; k < d; k++) {
                    q[i][k] -= dot * q[j][k];
                }
            }
            double norm = Math.sqrt(MlMatrices.dot(q[i], q[i]));
            for (int k = 0; k < d; k++) {
                q[i][k] /= norm;
            }
        }
        return q;
    }

    /** Orthogonal Procrustes {@code R = M (M^T M)^(-1/2)}, pseudo-inverse on a rank-deficient M. */
    private static double[][] procrustes(double[][] x, double[][] y) {
        int d = x[0].length;
        double[][] m = MlMatrices.multiply(MlMatrices.transpose(x), y);
        MlMatrices.Eigen e = MlMatrices.eigh(MlMatrices.multiply(MlMatrices.transpose(m), m));
        double[][] invRoot = new double[d][d];
        for (int k = 0; k < d; k++) {
            double s = e.values()[k];
            if (s <= 1e-9 * Math.max(1e-300, e.values()[0])) {
                continue;
            }
            double[] v = e.vector(k);
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    invRoot[i][j] += v[i] * v[j] / Math.sqrt(s);
                }
            }
        }
        return MlMatrices.multiply(m, invRoot);
    }

    /**
     * Embedding-space alignment.
     *
     * @param anchors the anchor counts for Procrustes
     * @param seed the seed
     * @return method to {@code {precision@1}}
     */
    public static Map<String, double[]> embeddingAlignment(int[] anchors, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 120;
        int d = 8;
        int clusters = 12;
        double[][] centres = new double[clusters][d];
        double[] scale = new double[d];
        for (int k = 0; k < d; k++) {
            scale[k] = Math.pow(0.75, k);
        }
        for (double[] c : centres) {
            for (int k = 0; k < d; k++) {
                c[k] = 2 * scale[k] * r.nextGaussian();
            }
        }
        double[][] x = new double[n][d];
        for (int i = 0; i < n; i++) {
            double[] c = centres[r.nextInt(clusters)];
            for (int k = 0; k < d; k++) {
                x[i][k] = c[k] + 0.4 * scale[k] * r.nextGaussian();
            }
        }
        double[][] rot = randomRotation(d, r);
        int[] p = permutation(n, r);
        double[][] y = new double[n][];
        for (int i = 0; i < n; i++) {
            double[] v = new double[d];
            for (int k = 0; k < d; k++) {
                for (int j = 0; j < d; j++) {
                    v[k] += x[i][j] * rot[j][k];
                }
                v[k] += 0.03 * r.nextGaussian();
            }
            y[p[i]] = v;
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        GromovWasserstein.Result gw = GromovWasserstein.align(GromovWasserstein.distances(x),
                GromovWasserstein.distances(y), 0.001, 40, 2, seed);
        out.put("GW (no anchors)", new double[] {accuracy(gw.matching(), p)});
        for (int k : anchors) {
            double[][] xa = new double[k][];
            double[][] ya = new double[k][];
            for (int i = 0; i < k; i++) {
                xa[i] = x[i];
                ya[i] = y[p[i]];
            }
            double[][] rk = procrustes(xa, ya);
            double[][] mapped = MlMatrices.multiply(x, rk);
            int[] nn = new int[n];
            for (int i = 0; i < n; i++) {
                double best = Double.POSITIVE_INFINITY;
                for (int j = 0; j < n; j++) {
                    double dist = MlMatrices.squaredDistance(mapped[i], y[j]);
                    if (dist < best) {
                        best = dist;
                        nn[i] = j;
                    }
                }
            }
            out.put("Procrustes k=" + k, new double[] {accuracy(nn, p)});
        }
        return out;
    }

    private static double entropy(double[] p) {
        double h = 0;
        for (double v : p) {
            h -= v > 0 ? v * Math.log(v) : 0;
        }
        return h;
    }

    private static double meanCost(double[] p, double[][] inputs, double[][] cost) {
        double[] a = floor(p);
        double s = 0;
        for (double[] h : inputs) {
            s += EntropicTransport.sinkhorn(a, floor(h), cost, 0.3, 500, 1e-7).transportCost() / inputs.length;
        }
        return s;
    }

    private static double[] floor(double[] p) {
        double[] q = new double[p.length];
        double t = 0;
        for (int i = 0; i < p.length; i++) {
            q[i] = Math.max(p[i], 1e-9);
            t += q[i];
        }
        for (int i = 0; i < p.length; i++) {
            q[i] /= t;
        }
        return q;
    }

    /**
     * Barycenters of one glyph class.
     *
     * @param shapeClass the {@code ShapeImages} class
     * @param count how many glyphs to average
     * @param seed the seed
     * @return method to {@code {mean transport cost to the inputs, entropy}}
     */
    public static Map<String, double[]> barycenter(int shapeClass, int count, long seed) {
        int side = 12;
        Tensor[] data = new ShapeImages(side, 0f).generate(count * ShapeImages.CLASSES * 3, seed);
        double[] img = data[0].toDoubleArray();
        double[] ids = data[1].toDoubleArray();
        int n = side * side;
        double[][] inputs = new double[count][n];
        int found = 0;
        for (int i = 0; i < ids.length && found < count; i++) {
            if ((int) ids[i] == shapeClass) {
                double t = 0;
                for (int j = 0; j < n; j++) {
                    inputs[found][j] = Math.max(0, img[i * n + j]);
                    t += inputs[found][j];
                }
                for (int j = 0; j < n; j++) {
                    inputs[found][j] /= t;
                }
                found++;
            }
        }
        if (found < count) {
            throw new IllegalStateException("not enough glyphs of class " + shapeClass);
        }
        double[][] cost = WassersteinBarycenter.gridCost(side);
        double[] w = EntropicTransport.uniform(count);
        double[] wb = WassersteinBarycenter.fixedSupport(inputs, w, cost, 0.3, 300);
        double[] mean = new double[n];
        for (double[] h : inputs) {
            for (int j = 0; j < n; j++) {
                mean[j] += h[j] / count;
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("Wasserstein barycenter", new double[] {meanCost(wb, inputs, cost), entropy(wb)});
        out.put("Euclidean mean", new double[] {meanCost(mean, inputs, cost), entropy(mean)});
        out.put("one input", new double[] {meanCost(inputs[0], inputs, cost), entropy(inputs[0])});
        return out;
    }
}
