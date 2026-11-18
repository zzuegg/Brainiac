package io.github.zzuegg.brainiac.experimental.circuits;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * A tensorised, smooth and decomposable probabilistic circuit on random region trees (RAT-SPNs,
 * Peharz et al., UAI 2019, <a href="https://arxiv.org/abs/1806.01910">arXiv:1806.01910</a>;
 * Einsum networks, Peharz et al., ICML 2020,
 * <a href="https://arxiv.org/abs/2004.06231">arXiv:2004.06231</a>).
 *
 * <p>Every region of each of {@code R} random {@link RegionTree trees} carries {@code K}
 * densities: a leaf region holds {@code K} univariate leaves (categorical, which covers
 * Bernoulli, or Gaussian), an inner region forms the {@code K x K} outer product of its
 * children's densities (product layer) and mixes it back to {@code K} (sum layer; 1 at the
 * root), and the replicas are mixed at the top. Because the circuit is smooth and decomposable,
 * every marginal is exact in one pass: a missing variable's leaves simply output 1. MPE
 * completion is the max-product pass with back-tracking.
 *
 * <p>Fitting is by EM, with the expected counts computed as circuit flows (the top-down
 * posterior of each edge, which equals the gradient of the log-likelihood with respect to the
 * log weights), or by Adam on softmax logits using the same flows as gradients. Missing values
 * in the training data are allowed: they contribute nothing to their leaves' statistics.
 */
public final class TensorizedCircuit {

    private final int[] categories;
    private final int units;
    private final RegionTree[] trees;
    private final double[] mix;
    private final double[][][][] theta;   // [r][node][k][c] for categorical leaves
    private final double[][][][] gauss;   // [r][node][k][{mean, var}] for Gaussian leaves
    private final double[][][][] weights; // [r][node][out][K*K]

    /**
     * Creates a circuit with random parameters.
     *
     * @param categories per variable: the number of categories, or 0 for a Gaussian leaf
     * @param units {@code K}, densities per region
     * @param replicas {@code R}, random trees mixed at the top
     * @param seed the seed
     */
    public TensorizedCircuit(int[] categories, int units, int replicas, long seed) {
        this.categories = categories.clone();
        this.units = units;
        int d = categories.length;
        SplittableRandom r = new SplittableRandom(seed);
        this.trees = new RegionTree[replicas];
        this.mix = new double[replicas];
        Arrays.fill(mix, 1.0 / replicas);
        this.theta = new double[replicas][][][];
        this.gauss = new double[replicas][][][];
        this.weights = new double[replicas][][][];
        for (int rep = 0; rep < replicas; rep++) {
            RegionTree t = RegionTree.random(d, seed * 31 + rep);
            trees[rep] = t;
            theta[rep] = new double[t.size()][][];
            gauss[rep] = new double[t.size()][][];
            weights[rep] = new double[t.size()][][];
            for (int node = 0; node < t.size(); node++) {
                if (t.isLeaf(node)) {
                    int c = categories[t.variable(node)];
                    if (c > 0) {
                        theta[rep][node] = new double[units][c];
                        for (double[] row : theta[rep][node]) {
                            double s = 0;
                            for (int k = 0; k < c; k++) {
                                row[k] = 0.5 + r.nextDouble();
                                s += row[k];
                            }
                            for (int k = 0; k < c; k++) {
                                row[k] /= s;
                            }
                        }
                    } else {
                        gauss[rep][node] = new double[units][];
                        for (int k = 0; k < units; k++) {
                            gauss[rep][node][k] = new double[] {r.nextGaussian(), 1};
                        }
                    }
                } else {
                    int out = node == t.root() ? 1 : units;
                    weights[rep][node] = new double[out][units * units];
                    for (double[] row : weights[rep][node]) {
                        double s = 0;
                        for (int q = 0; q < row.length; q++) {
                            row[q] = 0.5 + r.nextDouble();
                            s += row[q];
                        }
                        for (int q = 0; q < row.length; q++) {
                            row[q] /= s;
                        }
                    }
                }
            }
        }
    }

    /** @return the number of free parameters */
    public int parameterCount() {
        int p = mix.length - 1;
        for (int rep = 0; rep < trees.length; rep++) {
            RegionTree t = trees[rep];
            for (int node = 0; node < t.size(); node++) {
                if (t.isLeaf(node)) {
                    int c = categories[t.variable(node)];
                    p += units * (c > 0 ? c - 1 : 2);
                } else {
                    p += weights[rep][node].length * (units * units - 1);
                }
            }
        }
        return p;
    }

    private double leafLog(int rep, int node, int k, double value) {
        if (Double.isNaN(value)) {
            return 0;
        }
        int c = categories[trees[rep].variable(node)];
        if (c > 0) {
            return Math.log(Math.max(theta[rep][node][k][(int) value], 1e-300));
        }
        double[] g = gauss[rep][node][k];
        double z = value - g[0];
        return -0.5 * (Math.log(2 * Math.PI * g[1]) + z * z / g[1]);
    }

    /**
     * Upward pass: per node and unit, the log value. Inner nodes work in linear space relative to
     * each child's maximum, so the sum layer is a plain matrix-vector product.
     */
    private double[][] up(int rep, double[] x, boolean max, int[][] argmax) {
        RegionTree t = trees[rep];
        double[][] v = new double[t.size()][];
        int kk = units * units;
        double[] prod = new double[kk];
        for (int node = 0; node < t.size(); node++) {
            if (t.isLeaf(node)) {
                v[node] = new double[units];
                for (int k = 0; k < units; k++) {
                    v[node][k] = leafLog(rep, node, k, x[t.variable(node)]);
                }
                continue;
            }
            double[] a = v[t.left(node)];
            double[] b = v[t.right(node)];
            double ma = Double.NEGATIVE_INFINITY;
            double mb = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < units; k++) {
                ma = Math.max(ma, a[k]);
                mb = Math.max(mb, b[k]);
            }
            for (int i = 0; i < units; i++) {
                double ea = Math.exp(a[i] - ma);
                for (int j = 0; j < units; j++) {
                    prod[i * units + j] = ea * Math.exp(b[j] - mb);
                }
            }
            double[][] w = weights[rep][node];
            v[node] = new double[w.length];
            if (argmax != null) {
                argmax[node] = new int[w.length];
            }
            for (int o = 0; o < w.length; o++) {
                double[] wo = w[o];
                double s = 0;
                int arg = 0;
                double best = -1;
                for (int q = 0; q < kk; q++) {
                    double term = wo[q] * prod[q];
                    if (max) {
                        if (term > best) {
                            best = term;
                            arg = q;
                        }
                    } else {
                        s += term;
                    }
                }
                v[node][o] = Math.log(Math.max(max ? best : s, 1e-300)) + ma + mb;
                if (argmax != null) {
                    argmax[node][o] = arg;
                }
            }
        }
        return v;
    }

    /**
     * The exact log-likelihood (marginal over missing values).
     *
     * @param x a sample; {@code NaN} marks a missing (marginalised) variable; categorical values
     *     are category indices
     * @return {@code log p(x_observed)}
     */
    public double logLikelihood(double[] x) {
        double best = Double.NEGATIVE_INFINITY;
        double[] roots = new double[trees.length];
        for (int rep = 0; rep < trees.length; rep++) {
            roots[rep] = Math.log(mix[rep]) + up(rep, x, false, null)[trees[rep].root()][0];
            best = Math.max(best, roots[rep]);
        }
        double s = 0;
        for (double v : roots) {
            s += Math.exp(v - best);
        }
        return best + Math.log(s);
    }

    /**
     * The mean log-likelihood of a data set.
     *
     * @param x samples
     * @return the average
     */
    public double meanLogLikelihood(double[][] x) {
        double s = 0;
        for (double[] row : x) {
            s += logLikelihood(row);
        }
        return s / x.length;
    }

    /** Accumulates flows (expected counts) of one sample into the statistics. */
    private void flows(double[] x, double[] mixCount, double[][][][] wCount, double[][][][] tCount,
            double[][][][] gCount) {
        double[][][] vals = new double[trees.length][][];
        double best = Double.NEGATIVE_INFINITY;
        double[] roots = new double[trees.length];
        for (int rep = 0; rep < trees.length; rep++) {
            vals[rep] = up(rep, x, false, null);
            roots[rep] = Math.log(mix[rep]) + vals[rep][trees[rep].root()][0];
            best = Math.max(best, roots[rep]);
        }
        double total = 0;
        for (double v : roots) {
            total += Math.exp(v - best);
        }
        double logp = best + Math.log(total);
        for (int rep = 0; rep < trees.length; rep++) {
            RegionTree t = trees[rep];
            double post = Math.exp(roots[rep] - logp);
            mixCount[rep] += post;
            double[][] flow = new double[t.size()][];
            flow[t.root()] = new double[] {post};
            for (int node = t.root(); node >= 0; node--) {
                if (flow[node] == null) {
                    flow[node] = new double[units];
                }
                if (t.isLeaf(node)) {
                    double value = x[t.variable(node)];
                    if (Double.isNaN(value)) {
                        continue;
                    }
                    for (int k = 0; k < units; k++) {
                        if (categories[t.variable(node)] > 0) {
                            tCount[rep][node][k][(int) value] += flow[node][k];
                        } else {
                            gCount[rep][node][k][0] += flow[node][k];
                            gCount[rep][node][k][1] += flow[node][k] * value;
                            gCount[rep][node][k][2] += flow[node][k] * value * value;
                        }
                    }
                    continue;
                }
                int a = t.left(node);
                int b = t.right(node);
                if (flow[a] == null) {
                    flow[a] = new double[units];
                }
                if (flow[b] == null) {
                    flow[b] = new double[units];
                }
                double[][] w = weights[rep][node];
                double[] va = vals[rep][a];
                double[] vb = vals[rep][b];
                double[] ea = new double[units];
                double[] eb = new double[units];
                double ma = Double.NEGATIVE_INFINITY;
                double mb = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < units; k++) {
                    ma = Math.max(ma, va[k]);
                    mb = Math.max(mb, vb[k]);
                }
                for (int k = 0; k < units; k++) {
                    ea[k] = Math.exp(va[k] - ma);
                    eb[k] = Math.exp(vb[k] - mb);
                }
                for (int o = 0; o < w.length; o++) {
                    if (flow[node][o] == 0) {
                        continue;
                    }
                    // exp(va[i] + vb[j] - out) = ea[i] eb[j] exp(ma + mb - out)
                    double f = flow[node][o] * Math.exp(ma + mb - vals[rep][node][o]);
                    double[] wo = w[o];
                    double[] co = wCount[rep][node][o];
                    for (int i = 0; i < units; i++) {
                        double fi = f * ea[i];
                        for (int j = 0; j < units; j++) {
                            int q = i * units + j;
                            double e = fi * wo[q] * eb[j];
                            co[q] += e;
                            flow[a][i] += e;
                            flow[b][j] += e;
                        }
                    }
                }
            }
        }
    }

    private double[][][][] zerosLike(double[][][][] p, int extra) {
        double[][][][] z = new double[p.length][][][];
        for (int rep = 0; rep < p.length; rep++) {
            z[rep] = new double[p[rep].length][][];
            for (int node = 0; node < p[rep].length; node++) {
                if (p[rep][node] != null) {
                    z[rep][node] = new double[p[rep][node].length][extra > 0 ? extra : p[rep][node][0].length];
                }
            }
        }
        return z;
    }

    /**
     * Data-driven initialisation: k-means on the (mean-imputed) data gives {@code K} clusters;
     * unit {@code k} of every leaf takes cluster {@code k}'s smoothed marginal, and every sum
     * layer favours the "diagonal" product {@code (k, k)}, so the circuit starts as a
     * {@code K}-component mixture of product distributions (each replica from its own k-means
     * seed) and EM or Adam then refines the hierarchy.
     *
     * @param x training samples ({@code NaN} for missing)
     * @param seed the k-means seed
     * @return this
     */
    public TensorizedCircuit initialise(double[][] x, long seed) {
        int n = x.length;
        int d = categories.length;
        double[] mean = new double[d];
        int[] seen = new int[d];
        for (double[] row : x) {
            for (int v = 0; v < d; v++) {
                if (!Double.isNaN(row[v])) {
                    mean[v] += row[v];
                    seen[v]++;
                }
            }
        }
        for (int v = 0; v < d; v++) {
            mean[v] /= Math.max(1, seen[v]);
        }
        double[][] filled = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int v = 0; v < d; v++) {
                double value = x[i][v];
                filled[i][v] = Double.isNaN(value) ? mean[v] : categories[v] > 2 ? value / categories[v] : value;
            }
        }
        for (int rep = 0; rep < trees.length; rep++) {
            int[] lab = io.github.zzuegg.brainiac.experimental.ml.KMeans.fit(filled, units,
                    io.github.zzuegg.brainiac.experimental.ml.KMeans.Init.PLUS_PLUS, 1, 30, seed + rep).labels();
            double[] size = new double[units];
            for (int l : lab) {
                size[l]++;
            }
            RegionTree t = trees[rep];
            for (int node = 0; node < t.size(); node++) {
                if (t.isLeaf(node)) {
                    int v = t.variable(node);
                    for (int k = 0; k < units; k++) {
                        if (categories[v] > 0) {
                            double[] cnt = new double[categories[v]];
                            for (int i = 0; i < n; i++) {
                                if (lab[i] == k && !Double.isNaN(x[i][v])) {
                                    cnt[(int) x[i][v]]++;
                                }
                            }
                            normalise(theta[rep][node][k], cnt, 0.5);
                        } else {
                            double s = 0;
                            double s2 = 0;
                            double c = 0;
                            for (int i = 0; i < n; i++) {
                                if (lab[i] == k && !Double.isNaN(x[i][v])) {
                                    s += x[i][v];
                                    s2 += x[i][v] * x[i][v];
                                    c++;
                                }
                            }
                            if (c > 1) {
                                gauss[rep][node][k][0] = s / c;
                                gauss[rep][node][k][1] = Math.max(1e-2, s2 / c - (s / c) * (s / c));
                            }
                        }
                    }
                } else {
                    double[][] w = weights[rep][node];
                    for (int o = 0; o < w.length; o++) {
                        double[] cnt = new double[units * units];
                        if (w.length == 1) {
                            for (int k = 0; k < units; k++) {
                                cnt[k * units + k] = size[k];
                            }
                        } else {
                            cnt[o * units + o] = 4.0 * units * units;
                        }
                        normalise(w[o], cnt, 1);
                    }
                }
            }
        }
        return this;
    }

    /**
     * Fits by EM.
     *
     * @param x training samples ({@code NaN} for missing)
     * @param iterations EM iterations
     * @param prior a Dirichlet pseudo-count added to every weight and category, for example
     *     {@code 0.01}
     * @return this
     */
    public TensorizedCircuit fitEm(double[][] x, int iterations, double prior) {
        for (int it = 0; it < iterations; it++) {
            double[] mc = new double[mix.length];
            double[][][][] wc = zerosLike(weights, 0);
            double[][][][] tc = zerosLike(theta, 0);
            double[][][][] gc = zerosLike(gauss, 3);
            for (double[] row : x) {
                flows(row, mc, wc, tc, gc);
            }
            normalise(mix, mc, prior);
            for (int rep = 0; rep < trees.length; rep++) {
                for (int node = 0; node < trees[rep].size(); node++) {
                    if (weights[rep][node] != null) {
                        for (int o = 0; o < weights[rep][node].length; o++) {
                            normalise(weights[rep][node][o], wc[rep][node][o], prior);
                        }
                    }
                    if (theta[rep][node] != null) {
                        for (int k = 0; k < units; k++) {
                            normalise(theta[rep][node][k], tc[rep][node][k], prior);
                        }
                    }
                    if (gauss[rep][node] != null) {
                        for (int k = 0; k < units; k++) {
                            double[] g = gc[rep][node][k];
                            if (g[0] > 1e-9) {
                                double m = g[1] / g[0];
                                gauss[rep][node][k][0] = m;
                                gauss[rep][node][k][1] = Math.max(1e-3, g[2] / g[0] - m * m);
                            }
                        }
                    }
                }
            }
        }
        return this;
    }

    private static void normalise(double[] target, double[] counts, double prior) {
        double s = 0;
        for (double c : counts) {
            s += c + prior;
        }
        for (int q = 0; q < target.length; q++) {
            target[q] = (counts[q] + prior) / s;
        }
    }

    /**
     * Fits by mini-batch Adam on softmax logits of the sum weights and categorical leaves (the
     * flows are the gradients of the log-likelihood with respect to the log weights).
     *
     * @param x training samples
     * @param steps Adam steps
     * @param batch samples per step
     * @param lr the learning rate
     * @param seed the batch seed
     * @return this
     */
    public TensorizedCircuit fitAdam(double[][] x, int steps, int batch, double lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][][][] wm = zerosLike(weights, 0);
        double[][][][] wv = zerosLike(weights, 0);
        double[][][][] tm = zerosLike(theta, 0);
        double[][][][] tv = zerosLike(theta, 0);
        double[][][][] wl = logits(weights);
        double[][][][] tl = logits(theta);
        for (int s = 1; s <= steps; s++) {
            double[] mc = new double[mix.length];
            double[][][][] wc = zerosLike(weights, 0);
            double[][][][] tc = zerosLike(theta, 0);
            double[][][][] gc = zerosLike(gauss, 3);
            for (int b = 0; b < batch; b++) {
                flows(x[r.nextInt(x.length)], mc, wc, tc, gc);
            }
            adam(wl, wm, wv, wc, weights, s, lr, batch);
            adam(tl, tm, tv, tc, theta, s, lr, batch);
        }
        return this;
    }

    private static double[][][][] logits(double[][][][] p) {
        double[][][][] l = new double[p.length][][][];
        for (int rep = 0; rep < p.length; rep++) {
            l[rep] = new double[p[rep].length][][];
            for (int node = 0; node < p[rep].length; node++) {
                if (p[rep][node] != null) {
                    l[rep][node] = new double[p[rep][node].length][];
                    for (int o = 0; o < p[rep][node].length; o++) {
                        l[rep][node][o] = Arrays.stream(p[rep][node][o]).map(v -> Math.log(Math.max(v, 1e-12)))
                                .toArray();
                    }
                }
            }
        }
        return l;
    }

    private static void adam(double[][][][] logit, double[][][][] m, double[][][][] v, double[][][][] count,
            double[][][][] param, int step, double lr, int batch) {
        double b1 = 0.9;
        double b2 = 0.999;
        for (int rep = 0; rep < logit.length; rep++) {
            for (int node = 0; node < logit[rep].length; node++) {
                if (logit[rep][node] == null) {
                    continue;
                }
                for (int o = 0; o < logit[rep][node].length; o++) {
                    double[] l = logit[rep][node][o];
                    double[] c = count[rep][node][o];
                    double[] p = param[rep][node][o];
                    double total = 0;
                    for (double q : c) {
                        total += q;
                    }
                    double max = Double.NEGATIVE_INFINITY;
                    for (int q = 0; q < l.length; q++) {
                        double g = (c[q] - p[q] * total) / batch;   // ascent direction
                        m[rep][node][o][q] = b1 * m[rep][node][o][q] + (1 - b1) * g;
                        v[rep][node][o][q] = b2 * v[rep][node][o][q] + (1 - b2) * g * g;
                        double mh = m[rep][node][o][q] / (1 - Math.pow(b1, step));
                        double vh = v[rep][node][o][q] / (1 - Math.pow(b2, step));
                        l[q] += lr * mh / (Math.sqrt(vh) + 1e-8);
                        max = Math.max(max, l[q]);
                    }
                    double s = 0;
                    for (int q = 0; q < l.length; q++) {
                        p[q] = Math.exp(l[q] - max);
                        s += p[q];
                    }
                    for (int q = 0; q < l.length; q++) {
                        p[q] /= s;
                    }
                }
            }
        }
    }

    /**
     * MPE completion: the most probable joint state of the missing variables (of the
     * max-product circuit, the standard tractable approximation of MAP for non-deterministic
     * circuits).
     *
     * @param x a sample with {@code NaN} for missing values
     * @return a completed copy
     */
    public double[] mpe(double[] x) {
        int bestRep = 0;
        double best = Double.NEGATIVE_INFINITY;
        int[][][] arg = new int[trees.length][][];
        for (int rep = 0; rep < trees.length; rep++) {
            arg[rep] = new int[trees[rep].size()][];
            double v = Math.log(mix[rep]) + up(rep, x, true, arg[rep])[trees[rep].root()][0];
            if (v > best) {
                best = v;
                bestRep = rep;
            }
        }
        double[] out = x.clone();
        descend(bestRep, trees[bestRep].root(), 0, arg[bestRep], out);
        return out;
    }

    private void descend(int rep, int node, int unit, int[][] arg, double[] out) {
        RegionTree t = trees[rep];
        if (t.isLeaf(node)) {
            int v = t.variable(node);
            if (Double.isNaN(out[v])) {
                if (categories[v] > 0) {
                    double[] p = theta[rep][node][unit];
                    int b = 0;
                    for (int c = 1; c < p.length; c++) {
                        b = p[c] > p[b] ? c : b;
                    }
                    out[v] = b;
                } else {
                    out[v] = gauss[rep][node][unit][0];
                }
            }
            return;
        }
        int q = arg[node][unit];
        descend(rep, t.left(node), q / units, arg, out);
        descend(rep, t.right(node), q % units, arg, out);
    }

    /**
     * Classification by conditional inference: {@code argmax_c p(x_observed, label = c)}, with
     * every other missing input marginalised exactly.
     *
     * @param x a sample (the label entry is ignored)
     * @param label the label variable
     * @return the most probable category
     */
    public int classify(double[] x, int label) {
        double[] q = x.clone();
        int best = 0;
        double bestLl = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < categories[label]; c++) {
            q[label] = c;
            double ll = logLikelihood(q);
            if (ll > bestLl) {
                bestLl = ll;
                best = c;
            }
        }
        return best;
    }
}
