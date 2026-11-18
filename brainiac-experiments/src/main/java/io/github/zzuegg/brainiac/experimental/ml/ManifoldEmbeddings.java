package io.github.zzuegg.brainiac.experimental.ml;

import java.util.Arrays;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.SplittableRandom;

/**
 * Nonlinear dimensionality reduction by spectral manifold learning and by PaCMAP.
 *
 * <ul>
 *   <li>{@link Method#CLASSICAL_MDS}: Torgerson scaling of the double-centred squared Euclidean
 *       distances, which equals PCA;</li>
 *   <li>{@link Method#ISOMAP}: classical MDS of shortest-path distances on the symmetrised
 *       {@code k}-nearest-neighbour graph (Tenenbaum, de Silva and Langford, Science 2000);</li>
 *   <li>{@link Method#LLE}: locally linear embedding (Roweis and Saul, Science 2000), the bottom
 *       eigenvectors of {@code (I - W)^T (I - W)} for regularised local reconstruction weights
 *       {@code W};</li>
 *   <li>{@link Method#LAPLACIAN_EIGENMAPS}: the bottom nontrivial generalised eigenvectors of the
 *       heat-kernel {@code k}-NN graph Laplacian, {@code L y = lambda D y} (Belkin and Niyogi,
 *       Neural Computation 2003);</li>
 *   <li>{@link Method#DIFFUSION_MAPS}: the top nontrivial eigenvectors of the
 *       density-normalised ({@code alpha = 1}) Gaussian diffusion operator scaled by their
 *       eigenvalues, so Euclidean distance approximates diffusion distance at {@code t = 1}
 *       (Coifman and Lafon, ACHA 2006);</li>
 *   <li>{@link Method#PACMAP}: pairwise controlled manifold approximation (Wang, Huang, Rudin and
 *       Shaposhnik, JMLR 2021, <a href="https://arxiv.org/abs/2012.04456">arXiv:2012.04456</a>):
 *       neighbour, mid-near and further pairs with losses {@code d/(10+d)},
 *       {@code d/(10000+d)} and {@code 1/(1+d)} on {@code d = 1 + |y_i - y_j|^2}, weighted by a
 *       three-phase schedule over 450 Adam iterations from a scaled PCA start.</li>
 * </ul>
 * The few extreme eigenpairs each method needs come from Lanczos iterations on the dense
 * {@code n x n} operator, so these suit up to a few thousand points.
 */
public final class ManifoldEmbeddings {

    /** The embedding method. */
    public enum Method {
        /** Classical (Torgerson) multidimensional scaling. */
        CLASSICAL_MDS,
        /** MDS of k-NN graph geodesics. */
        ISOMAP,
        /** Locally linear embedding. */
        LLE,
        /** Laplacian eigenmaps. */
        LAPLACIAN_EIGENMAPS,
        /** Diffusion maps. */
        DIFFUSION_MAPS,
        /** PaCMAP. */
        PACMAP
    }

    private ManifoldEmbeddings() {
    }

    /**
     * Embeds points.
     *
     * @param x the points {@code [n][D]}
     * @param method the method
     * @param dimensions the embedding dimension
     * @param neighbours the neighbourhood size {@code k} (graph methods and PaCMAP)
     * @param seed seeds PaCMAP's pair sampling
     * @return the embedding {@code [n][dimensions]}
     */
    public static double[][] embed(double[][] x, Method method, int dimensions, int neighbours, long seed) {
        Objects.requireNonNull(method, "method");
        return switch (method) {
            case CLASSICAL_MDS -> mds(MlMatrices.squaredDistances(x), dimensions);
            case ISOMAP -> {
                double[][] g = geodesicDistances(x, neighbours);
                for (double[] row : g) {
                    for (int j = 0; j < row.length; j++) {
                        row[j] *= row[j];
                    }
                }
                yield mds(g, dimensions);
            }
            case LLE -> lle(x, dimensions, neighbours);
            case LAPLACIAN_EIGENMAPS -> laplacian(x, dimensions, neighbours);
            case DIFFUSION_MAPS -> diffusion(x, dimensions, neighbours);
            case PACMAP -> pacmap(x, dimensions, neighbours, seed);
        };
    }

    /** Torgerson scaling of squared distances. */
    private static double[][] mds(double[][] d2, int dims) {
        int n = d2.length;
        double[] row = new double[n];
        double all = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                row[i] += d2[i][j] / n;
            }
            all += row[i] / n;
        }
        double[][] b = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                b[i][j] = -0.5 * (d2[i][j] - row[i] - row[j] + all);
            }
        }
        MlMatrices.Eigen e = extremeEigen(b, dims, true);
        double[][] y = new double[n][dims];
        for (int c = 0; c < dims; c++) {
            double s = Math.sqrt(Math.max(e.values()[c], 0));
            for (int i = 0; i < n; i++) {
                y[i][c] = e.vectors()[i][c] * s;
            }
        }
        return y;
    }

    /**
     * The {@code count} largest (or smallest) eigenpairs of a symmetric matrix by Lanczos with full
     * re-orthogonalisation and implicit QL on the tridiagonal, which is far cheaper than a dense
     * eigendecomposition when only a few pairs are needed.
     *
     * @param a the symmetric matrix {@code [n][n]}
     * @param count the number of pairs
     * @param largest the largest when true, else the smallest (in ascending order)
     * @return eigenvalues and eigenvectors (columns) in the requested order
     */
    static MlMatrices.Eigen extremeEigen(double[][] a, int count, boolean largest) {
        int n = a.length;
        int m = Math.min(n, Math.max(10 * count, 150));
        SplittableRandom r = new SplittableRandom(17);
        double[][] q = new double[m][];
        double[] alpha = new double[m];
        double[] beta = new double[m];
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
            v[i] = r.nextDouble() - 0.5;
        }
        normalise(v);
        int used = 0;
        for (int k = 0; k < m; k++) {
            q[k] = v;
            used = k + 1;
            double[] w = new double[n];
            for (int i = 0; i < n; i++) {
                double s = 0;
                double[] row = a[i];
                for (int j = 0; j < n; j++) {
                    s += row[j] * v[j];
                }
                w[i] = s;
            }
            alpha[k] = MlMatrices.dot(w, v);
            for (int pass = 0; pass < 2; pass++) {
                for (int b = 0; b <= k; b++) {
                    double d = MlMatrices.dot(w, q[b]);
                    for (int i = 0; i < n; i++) {
                        w[i] -= d * q[b][i];
                    }
                }
            }
            double bt = Math.sqrt(MlMatrices.dot(w, w));
            if (k == m - 1 || bt < 1e-10) {
                break;
            }
            beta[k] = bt;
            for (int i = 0; i < n; i++) {
                w[i] /= bt;
            }
            v = w;
        }
        double[] d = Arrays.copyOf(alpha, used);
        double[] e = new double[used];
        for (int k = 0; k + 1 < used; k++) {
            e[k] = beta[k];
        }
        double[][] z = new double[used][used];
        for (int k = 0; k < used; k++) {
            z[k][k] = 1;
        }
        tridiagonalQl(d, e, z);
        Integer[] order = new Integer[used];
        for (int k = 0; k < used; k++) {
            order[k] = k;
        }
        Arrays.sort(order, (p, s) -> largest ? Double.compare(d[s], d[p]) : Double.compare(d[p], d[s]));
        int c = Math.min(count, used);
        double[] values = new double[c];
        double[][] vectors = new double[n][c];
        for (int t = 0; t < c; t++) {
            int col = order[t];
            values[t] = d[col];
            for (int k = 0; k < used; k++) {
                double coef = z[k][col];
                for (int i = 0; i < n; i++) {
                    vectors[i][t] += coef * q[k][i];
                }
            }
        }
        return new MlMatrices.Eigen(values, vectors);
    }

    private static void normalise(double[] v) {
        double s = Math.sqrt(MlMatrices.dot(v, v));
        for (int i = 0; i < v.length; i++) {
            v[i] /= s;
        }
    }

    /**
     * Implicit QL with Wilkinson shifts on a symmetric tridiagonal matrix (diagonal {@code d},
     * sub-diagonal {@code e[0..n-2]}), accumulating rotations into {@code z}.
     */
    private static void tridiagonalQl(double[] d, double[] e, double[][] z) {
        int n = d.length;
        for (int l = 0; l < n; l++) {
            int iter = 0;
            int m;
            do {
                for (m = l; m < n - 1; m++) {
                    double dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
                    if (Math.abs(e[m]) <= 1e-15 * dd) {
                        break;
                    }
                }
                if (m != l) {
                    if (iter++ == 60) {
                        break;
                    }
                    double g = (d[l + 1] - d[l]) / (2 * e[l]);
                    double r = Math.hypot(g, 1);
                    g = d[m] - d[l] + e[l] / (g + Math.copySign(r, g));
                    double s = 1;
                    double c = 1;
                    double p = 0;
                    int i;
                    for (i = m - 1; i >= l; i--) {
                        double f = s * e[i];
                        double b = c * e[i];
                        r = Math.hypot(f, g);
                        e[i + 1] = r;
                        if (r == 0) {
                            d[i + 1] -= p;
                            e[m] = 0;
                            break;
                        }
                        s = f / r;
                        c = g / r;
                        g = d[i + 1] - p;
                        r = (d[i] - g) * s + 2 * c * b;
                        p = s * r;
                        d[i + 1] = g + p;
                        g = c * r - b;
                        for (int k = 0; k < n; k++) {
                            f = z[k][i + 1];
                            z[k][i + 1] = s * z[k][i] + c * f;
                            z[k][i] = c * z[k][i] - s * f;
                        }
                    }
                    if (r == 0 && i >= l) {
                        continue;
                    }
                    d[l] -= p;
                    e[l] = g;
                    e[m] = 0;
                }
            } while (m != l);
        }
    }

    private static int[][] knn(double[][] d2, int k) {
        int n = d2.length;
        int[][] nn = new int[n][];
        for (int i = 0; i < n; i++) {
            nn[i] = MlMetrics.nearest(d2[i], i, k);
        }
        return nn;
    }

    /**
     * Shortest-path distances on the symmetrised {@code k}-NN graph with Euclidean edge lengths
     * (Dijkstra from every node). Pairs in different components get 1.5 times the largest finite
     * distance.
     *
     * @param x the points
     * @param k the neighbourhood size
     * @return {@code [n][n]} geodesic distances
     */
    public static double[][] geodesicDistances(double[][] x, int k) {
        int n = x.length;
        double[][] d2 = MlMatrices.squaredDistances(x);
        int[][] nn = knn(d2, k);
        int[][] adj = new int[n][];
        int[] deg = new int[n];
        boolean[][] edge = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j : nn[i]) {
                edge[i][j] = true;
                edge[j][i] = true;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (edge[i][j]) {
                    deg[i]++;
                }
            }
            adj[i] = new int[deg[i]];
            int c = 0;
            for (int j = 0; j < n; j++) {
                if (edge[i][j]) {
                    adj[i][c++] = j;
                }
            }
        }
        double[][] g = new double[n][];
        double maxFinite = 0;
        for (int s = 0; s < n; s++) {
            double[] dist = new double[n];
            Arrays.fill(dist, Double.POSITIVE_INFINITY);
            dist[s] = 0;
            PriorityQueue<double[]> pq = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
            pq.add(new double[] {0, s});
            while (!pq.isEmpty()) {
                double[] top = pq.poll();
                int u = (int) top[1];
                if (top[0] > dist[u]) {
                    continue;
                }
                for (int v : adj[u]) {
                    double nd = dist[u] + Math.sqrt(d2[u][v]);
                    if (nd < dist[v]) {
                        dist[v] = nd;
                        pq.add(new double[] {nd, v});
                    }
                }
            }
            for (double v : dist) {
                if (Double.isFinite(v)) {
                    maxFinite = Math.max(maxFinite, v);
                }
            }
            g[s] = dist;
        }
        for (double[] row : g) {
            for (int j = 0; j < n; j++) {
                if (!Double.isFinite(row[j])) {
                    row[j] = 1.5 * maxFinite;
                }
            }
        }
        return g;
    }

    private static double[][] lle(double[][] x, int dims, int k) {
        int n = x.length;
        int d = x[0].length;
        int[][] nn = knn(MlMatrices.squaredDistances(x), k);
        double[][] w = new double[n][n];
        for (int i = 0; i < n; i++) {
            double[][] z = new double[k][d];
            for (int a = 0; a < k; a++) {
                for (int j = 0; j < d; j++) {
                    z[a][j] = x[nn[i][a]][j] - x[i][j];
                }
            }
            double[][] c = new double[k][k];
            double trace = 0;
            for (int a = 0; a < k; a++) {
                for (int b = 0; b < k; b++) {
                    c[a][b] = MlMatrices.dot(z[a], z[b]);
                }
                trace += c[a][a];
            }
            double reg = 1e-3 * Math.max(trace, 1e-12);
            for (int a = 0; a < k; a++) {
                c[a][a] += reg;
            }
            double[] ones = new double[k];
            Arrays.fill(ones, 1);
            double[] wi = MlMatrices.choleskySolve(MlMatrices.cholesky(c), ones);
            double s = 0;
            for (double v : wi) {
                s += v;
            }
            for (int a = 0; a < k; a++) {
                w[i][nn[i][a]] = wi[a] / s;
            }
        }
        // M = (I - W)^T (I - W)
        double[][] iw = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                iw[i][j] = (i == j ? 1 : 0) - w[i][j];
            }
        }
        double[][] m = new double[n][n];
        for (int r = 0; r < n; r++) {
            for (int i = 0; i < n; i++) {
                double v = iw[r][i];
                if (v == 0) {
                    continue;
                }
                for (int j = 0; j < n; j++) {
                    m[i][j] += v * iw[r][j];
                }
            }
        }
        // shift-invert: the smallest eigenvalues of M are the largest of (M + sigma I)^-1, where
        // Lanczos converges in a few dozen steps
        double sigma = 1e-3;
        for (int i = 0; i < n; i++) {
            m[i][i] += sigma;
        }
        double[][] inv = MlMatrices.choleskyInverse(MlMatrices.cholesky(m));
        MlMatrices.Eigen e = extremeEigen(inv, dims + 1, true);
        double[][] y = new double[n][dims];
        for (int c = 0; c < dims; c++) {
            // skip the constant eigenvector (smallest eigenvalue of M)
            int col = c + 1;
            for (int i = 0; i < n; i++) {
                y[i][c] = e.vectors()[i][col] * Math.sqrt(n);
            }
        }
        return y;
    }

    /** Symmetric heat-kernel k-NN affinities with {@code t} the median squared neighbour distance. */
    private static double[][] heatGraph(double[][] x, int k) {
        int n = x.length;
        double[][] d2 = MlMatrices.squaredDistances(x);
        int[][] nn = knn(d2, k);
        double[] all = new double[n * k];
        int c = 0;
        for (int i = 0; i < n; i++) {
            for (int j : nn[i]) {
                all[c++] = d2[i][j];
            }
        }
        Arrays.sort(all);
        double t = Math.max(all[all.length / 2], 1e-12);
        double[][] w = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j : nn[i]) {
                double v = Math.exp(-d2[i][j] / t);
                w[i][j] = v;
                w[j][i] = v;
            }
        }
        return w;
    }

    private static double[][] laplacian(double[][] x, int dims, int k) {
        int n = x.length;
        double[][] w = heatGraph(x, k);
        double[] deg = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                deg[i] += w[i][j];
            }
        }
        // top eigenvectors of D^-1/2 W D^-1/2 are the bottom ones of the normalised Laplacian
        double[][] s = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                s[i][j] = w[i][j] / Math.sqrt(deg[i] * deg[j] + 1e-300);
            }
        }
        MlMatrices.Eigen e = extremeEigen(s, dims + 1, true);
        double[][] y = new double[n][dims];
        for (int c = 0; c < dims; c++) {
            for (int i = 0; i < n; i++) {
                y[i][c] = e.vectors()[i][c + 1] / Math.sqrt(deg[i] + 1e-300);
            }
        }
        return y;
    }

    private static double[][] diffusion(double[][] x, int dims, int k) {
        int n = x.length;
        double[][] d2 = MlMatrices.squaredDistances(x);
        int[][] nn = knn(d2, k);
        double[] kth = new double[n];
        for (int i = 0; i < n; i++) {
            kth[i] = d2[i][nn[i][k - 1]];
        }
        double[] sorted = kth.clone();
        Arrays.sort(sorted);
        double eps = Math.max(sorted[n / 2], 1e-12);
        double[][] kern = new double[n][n];
        double[] q = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                kern[i][j] = Math.exp(-d2[i][j] / eps);
                q[i] += kern[i][j];
            }
        }
        // alpha = 1 normalisation removes the sampling density
        double[] deg = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                kern[i][j] /= q[i] * q[j];
                deg[i] += kern[i][j];
            }
        }
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = kern[i][j] / Math.sqrt(deg[i] * deg[j]);
            }
        }
        MlMatrices.Eigen e = extremeEigen(a, dims + 1, true);
        double[][] y = new double[n][dims];
        for (int c = 0; c < dims; c++) {
            double lambda = e.values()[c + 1];
            for (int i = 0; i < n; i++) {
                y[i][c] = lambda * e.vectors()[i][c + 1] / Math.sqrt(deg[i]);
            }
        }
        return y;
    }

    private static double[][] pacmap(double[][] x, int dims, int k, long seed) {
        int n = x.length;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] d2 = MlMatrices.squaredDistances(x);
        int extra = Math.min(k + 50, n - 1);
        int[][] nn = knn(d2, extra);
        // scaled distances d_ij^2 / (sigma_i sigma_j), sigma = mean distance to the 4th-6th neighbours
        double[] sigma = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int a = 3; a < 6 && a < extra; a++) {
                s += Math.sqrt(d2[i][nn[i][a]]) / 3;
            }
            sigma[i] = Math.max(s, 1e-12);
        }
        int nNb = k;
        int nMn = Math.max(1, k / 2);
        int nFp = 2 * k;
        int[][] nb = new int[n][nNb];
        for (int i = 0; i < n; i++) {
            Integer[] idx = new Integer[extra];
            for (int a = 0; a < extra; a++) {
                idx[a] = nn[i][a];
            }
            final int ii = i;
            Arrays.sort(idx, (a, b) -> Double.compare(d2[ii][a] / (sigma[ii] * sigma[a]),
                    d2[ii][b] / (sigma[ii] * sigma[b])));
            for (int a = 0; a < nNb; a++) {
                nb[i][a] = idx[a];
            }
        }
        int[][] mn = new int[n][nMn];
        int[][] fp = new int[n][nFp];
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < nMn; a++) {
                // second closest of six random points
                int best = -1;
                int second = -1;
                for (int s = 0; s < 6; s++) {
                    int j = r.nextInt(n);
                    if (j == i) {
                        continue;
                    }
                    if (best < 0 || d2[i][j] < d2[i][best]) {
                        second = best;
                        best = j;
                    } else if (second < 0 || d2[i][j] < d2[i][second]) {
                        second = j;
                    }
                }
                mn[i][a] = second >= 0 ? second : (best >= 0 ? best : (i + 1) % n);
            }
            for (int a = 0; a < nFp; a++) {
                int j;
                do {
                    j = r.nextInt(n);
                } while (j == i);
                fp[i][a] = j;
            }
        }
        double[][] y = ComponentAnalysis.pca(x, dims).transform(x);
        double sd = 0;
        for (double[] p : y) {
            sd += p[0] * p[0] / n;
        }
        sd = Math.sqrt(Math.max(sd, 1e-24));
        for (double[] p : y) {
            for (int c = 0; c < dims; c++) {
                p[c] = p[c] / sd * 0.01;
            }
        }
        double[][] m = new double[n][dims];
        double[][] v = new double[n][dims];
        double lr = 1.0;
        int iterations = 450;
        for (int t = 1; t <= iterations; t++) {
            double wNb;
            double wMn;
            double wFp = 1;
            if (t <= 100) {
                wNb = 2;
                wMn = 1000 * (1 - (t - 1) / 100.0) + 3 * ((t - 1) / 100.0);
            } else if (t <= 200) {
                wNb = 3;
                wMn = 3;
            } else {
                wNb = 1;
                wMn = 0;
            }
            double[][] g = new double[n][dims];
            for (int i = 0; i < n; i++) {
                for (int j : nb[i]) {
                    pairGradient(y, i, j, g, wNb * 10, 10, true);
                }
                if (wMn > 0) {
                    for (int j : mn[i]) {
                        pairGradient(y, i, j, g, wMn * 10000, 10000, true);
                    }
                }
                for (int j : fp[i]) {
                    pairGradient(y, i, j, g, wFp, 1, false);
                }
            }
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < dims; c++) {
                    m[i][c] = 0.9 * m[i][c] + 0.1 * g[i][c];
                    v[i][c] = 0.999 * v[i][c] + 0.001 * g[i][c] * g[i][c];
                    y[i][c] -= lr * (m[i][c] / (1 - Math.pow(0.9, t)))
                            / (Math.sqrt(v[i][c] / (1 - Math.pow(0.999, t))) + 1e-7);
                }
            }
        }
        return y;
    }

    /**
     * Adds the gradient of one pair loss: attractive {@code w d/(c+d)} has
     * {@code dl/dd = w c/(c+d)^2}, repulsive {@code w/(1+d)} has {@code dl/dd = -w/(1+d)^2}, with
     * {@code d = 1 + |y_i - y_j|^2}.
     */
    private static void pairGradient(double[][] y, int i, int j, double[][] g, double scale, double c,
            boolean attractive) {
        int dims = y[i].length;
        double dd = 1;
        for (int a = 0; a < dims; a++) {
            double diff = y[i][a] - y[j][a];
            dd += diff * diff;
        }
        double coef = attractive ? scale / ((c + dd) * (c + dd)) : -scale / ((1 + dd) * (1 + dd));
        for (int a = 0; a < dims; a++) {
            double diff = 2 * (y[i][a] - y[j][a]) * coef;
            g[i][a] += diff;
            g[j][a] -= diff;
        }
    }

    /**
     * Continuity (Venna and Kaski 2001): trustworthiness with the two spaces exchanged, so it
     * penalises original neighbours that the embedding pulls apart.
     *
     * @param original the points
     * @param embedded their embedding
     * @param k the neighbourhood size
     * @return the continuity in {@code [0, 1]}
     */
    public static double continuity(double[][] original, double[][] embedded, int k) {
        return MlMetrics.trustworthiness(embedded, original, k);
    }

    /**
     * Spearman rank correlation of two distance matrices over random pairs.
     *
     * @param reference reference distances {@code [n][n]}
     * @param embedded the embedding {@code [n][e]}
     * @param pairs the number of sampled pairs
     * @param seed the sampling seed
     * @return the rank correlation
     */
    public static double spearman(double[][] reference, double[][] embedded, int pairs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = embedded.length;
        double[] a = new double[pairs];
        double[] b = new double[pairs];
        for (int p = 0; p < pairs; p++) {
            int i = r.nextInt(n);
            int j;
            do {
                j = r.nextInt(n);
            } while (j == i);
            a[p] = reference[i][j];
            b[p] = Math.sqrt(MlMatrices.squaredDistance(embedded[i], embedded[j]));
        }
        double[] ra = ranks(a);
        double[] rb = ranks(b);
        double ma = (pairs - 1) / 2.0;
        double num = 0;
        double da = 0;
        double db = 0;
        for (int p = 0; p < pairs; p++) {
            num += (ra[p] - ma) * (rb[p] - ma);
            da += (ra[p] - ma) * (ra[p] - ma);
            db += (rb[p] - ma) * (rb[p] - ma);
        }
        return num / Math.sqrt(da * db);
    }

    private static double[] ranks(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(v[p], v[q]));
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[idx[i]] = i;
        }
        return r;
    }
}
