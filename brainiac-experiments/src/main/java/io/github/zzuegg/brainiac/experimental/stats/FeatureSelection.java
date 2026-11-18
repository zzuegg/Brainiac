package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Feature selection for binary classification, each method giving a score per feature
 * (higher = more relevant).
 *
 * <ul>
 *   <li>{@link Method#MRMR} (Peng, Long and Ding, TPAMI 2005): greedy maximum relevance minus
 *       mean redundancy, with mutual information from quantile-binned features; the score is the
 *       reverse selection order.</li>
 *   <li>{@link Method#RELIEFF} (Kononenko, ECML 1994): feature weights move by the differences
 *       to nearest misses minus nearest hits, which catches interactions such as XOR pairs.</li>
 *   <li>{@link Method#BORUTA} (Kursa and Rudnicki, J. Stat. Softw. 2010): a random forest's
 *       held-out permutation importance against shuffled "shadow" copies; the score is the share
 *       of rounds a feature beats the best shadow.</li>
 *   <li>{@link Method#STOCHASTIC_GATES} (Yamada et al., ICML 2020,
 *       <a href="https://arxiv.org/abs/1810.04247">arXiv:1810.04247</a>): an MLP on gated inputs
 *       {@code x * clamp(mu + sigma eps, 0, 1)} with the relaxed l0 penalty
 *       {@code lambda sum Phi(mu / sigma)}; the score is {@code mu}.</li>
 *   <li>{@link Method#LASSONET} (Lemhadri et al., JMLR 2021,
 *       <a href="https://arxiv.org/abs/1907.12207">arXiv:1907.12207</a>): a residual MLP
 *       {@code theta'x + g(x)} whose first-layer weights of feature {@code j} are bounded by
 *       {@code M |theta_j|}, trained by proximal SGD with the hierarchical proximal operator
 *       along an increasing lambda path; the score is the lambda at which a feature leaves.</li>
 * </ul>
 */
public final class FeatureSelection {

    /** The method. */
    public enum Method {
        /** Minimum redundancy maximum relevance. */
        MRMR,
        /** ReliefF. */
        RELIEFF,
        /** Boruta. */
        BORUTA,
        /** Stochastic gates. */
        STOCHASTIC_GATES,
        /** LassoNet. */
        LASSONET
    }

    private FeatureSelection() {
    }

    /**
     * Scores features.
     *
     * @param method the method
     * @param x rows {@code [n][p]}
     * @param y labels in {0, 1}
     * @param seed the seed
     * @return a score per feature
     */
    public static double[] scores(Method method, double[][] x, int[] y, long seed) {
        return switch (method) {
            case MRMR -> mrmr(x, y, x[0].length);
            case RELIEFF -> reliefF(x, y, 10, Math.min(x.length, 300), seed);
            case BORUTA -> boruta(x, y, 8, seed);
            case STOCHASTIC_GATES -> stochasticGates(x, y, 0.08, 400, seed);
            case LASSONET -> lassoNet(x, y, 10, seed);
        };
    }

    /**
     * The indices of the {@code k} highest scores.
     *
     * @param scores scores
     * @param k how many
     * @return the indices, best first
     */
    public static int[] top(double[] scores, int k) {
        Integer[] o = new Integer[scores.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = i;
        }
        Arrays.sort(o, (a, b) -> Double.compare(scores[b], scores[a]));
        int[] out = new int[Math.min(k, o.length)];
        for (int i = 0; i < out.length; i++) {
            out[i] = o[i];
        }
        return out;
    }

    // ------------------------------------------------------------------ mRMR

    private static int[][] binned(double[][] x, int bins) {
        int n = x.length;
        int p = x[0].length;
        int[][] b = new int[p][n];
        for (int j = 0; j < p; j++) {
            double[] col = new double[n];
            for (int i = 0; i < n; i++) {
                col[i] = x[i][j];
            }
            double[] s = col.clone();
            Arrays.sort(s);
            double[] cuts = new double[bins - 1];
            for (int c = 0; c < bins - 1; c++) {
                cuts[c] = s[(int) ((c + 1.0) / bins * (n - 1))];
            }
            for (int i = 0; i < n; i++) {
                int k = 0;
                while (k < cuts.length && col[i] > cuts[k]) {
                    k++;
                }
                b[j][i] = k;
            }
        }
        return b;
    }

    /**
     * Mutual information (nats) between two discrete codes.
     *
     * @param a codes in {@code 0 .. ka-1}
     * @param ka the size of a's alphabet
     * @param b codes in {@code 0 .. kb-1}
     * @param kb the size of b's alphabet
     * @return the plug-in mutual information
     */
    public static double mutualInformation(int[] a, int ka, int[] b, int kb) {
        int n = a.length;
        double[][] joint = new double[ka][kb];
        double[] pa = new double[ka];
        double[] pb = new double[kb];
        for (int i = 0; i < n; i++) {
            joint[a[i]][b[i]] += 1.0 / n;
            pa[a[i]] += 1.0 / n;
            pb[b[i]] += 1.0 / n;
        }
        double mi = 0;
        for (int u = 0; u < ka; u++) {
            for (int v = 0; v < kb; v++) {
                if (joint[u][v] > 0) {
                    mi += joint[u][v] * Math.log(joint[u][v] / (pa[u] * pb[v]));
                }
            }
        }
        return mi;
    }

    private static double[] mrmr(double[][] x, int[] y, int k) {
        int p = x[0].length;
        int bins = 8;
        int[][] b = binned(x, bins);
        double[] rel = new double[p];
        for (int j = 0; j < p; j++) {
            rel[j] = mutualInformation(b[j], bins, y, 2);
        }
        boolean[] chosen = new boolean[p];
        double[] redundancy = new double[p];
        double[] score = new double[p];
        for (int step = 0; step < Math.min(k, p); step++) {
            int best = -1;
            double bestV = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < p; j++) {
                if (chosen[j]) {
                    continue;
                }
                double v = rel[j] - (step == 0 ? 0 : redundancy[j] / step);
                if (v > bestV) {
                    bestV = v;
                    best = j;
                }
            }
            chosen[best] = true;
            score[best] = p - step;
            for (int j = 0; j < p; j++) {
                if (!chosen[j]) {
                    redundancy[j] += mutualInformation(b[j], bins, b[best], bins);
                }
            }
        }
        return score;
    }

    // ------------------------------------------------------------------ ReliefF

    private static double[] reliefF(double[][] x, int[] y, int k, int samples, long seed) {
        int n = x.length;
        int p = x[0].length;
        double[] lo = new double[p];
        double[] range = new double[p];
        for (int j = 0; j < p; j++) {
            double mn = Double.POSITIVE_INFINITY;
            double mx = Double.NEGATIVE_INFINITY;
            for (double[] r : x) {
                mn = Math.min(mn, r[j]);
                mx = Math.max(mx, r[j]);
            }
            lo[j] = mn;
            range[j] = Math.max(mx - mn, 1e-12);
        }
        double[] w = new double[p];
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < samples; s++) {
            int i = r.nextInt(n);
            for (int cls = 0; cls <= 1; cls++) {
                double[] best = new double[k];
                int[] idx = new int[k];
                Arrays.fill(best, Double.POSITIVE_INFINITY);
                for (int q = 0; q < n; q++) {
                    if (q == i || y[q] != cls) {
                        continue;
                    }
                    double d = 0;
                    for (int j = 0; j < p; j++) {
                        d += Math.abs(x[i][j] - x[q][j]) / range[j];
                    }
                    if (d < best[k - 1]) {
                        int a = k - 1;
                        while (a > 0 && best[a - 1] > d) {
                            best[a] = best[a - 1];
                            idx[a] = idx[a - 1];
                            a--;
                        }
                        best[a] = d;
                        idx[a] = q;
                    }
                }
                double sign = cls == y[i] ? -1 : 1;
                for (int a = 0; a < k; a++) {
                    if (Double.isInfinite(best[a])) {
                        continue;
                    }
                    for (int j = 0; j < p; j++) {
                        w[j] += sign * Math.abs(x[i][j] - x[idx[a]][j]) / range[j] / (samples * k);
                    }
                }
            }
        }
        return w;
    }

    // ------------------------------------------------------------------ Boruta

    /**
     * Held-out permutation importance of a random forest: the accuracy drop when one column of a
     * 30% validation split is shuffled.
     *
     * @param x rows
     * @param y labels
     * @param trees trees
     * @param seed the seed
     * @return the importance per feature
     */
    public static double[] permutationImportance(double[][] x, int[] y, int trees, long seed) {
        int n = x.length;
        int p = x[0].length;
        int cut = (int) (0.7 * n);
        double[][] xt = Arrays.copyOfRange(x, 0, cut);
        int[] yt = Arrays.copyOfRange(y, 0, cut);
        double[][] xv = new double[n - cut][];
        for (int i = cut; i < n; i++) {
            xv[i - cut] = x[i].clone();
        }
        int[] yv = Arrays.copyOfRange(y, cut, n);
        RandomForest f = RandomForest.fit(xt, yt, 2, new RandomForest.Settings(RandomForest.Mode.RANDOM_FOREST,
                trees, 12, 2, 0, seed));
        double base = accuracy(f.predict(xv), yv);
        SplittableRandom r = new SplittableRandom(seed);
        double[] imp = new double[p];
        for (int j = 0; j < p; j++) {
            double[] keep = new double[xv.length];
            for (int i = 0; i < xv.length; i++) {
                keep[i] = xv[i][j];
            }
            for (int i = xv.length - 1; i > 0; i--) {
                int a = r.nextInt(i + 1);
                double t = xv[i][j];
                xv[i][j] = xv[a][j];
                xv[a][j] = t;
            }
            imp[j] = base - accuracy(f.predict(xv), yv);
            for (int i = 0; i < xv.length; i++) {
                xv[i][j] = keep[i];
            }
        }
        return imp;
    }

    private static double accuracy(int[] pred, int[] y) {
        int c = 0;
        for (int i = 0; i < y.length; i++) {
            if (pred[i] == y[i]) {
                c++;
            }
        }
        return (double) c / y.length;
    }

    private static double[] boruta(double[][] x, int[] y, int rounds, long seed) {
        int n = x.length;
        int p = x[0].length;
        double[] hits = new double[p];
        SplittableRandom r = new SplittableRandom(seed);
        for (int round = 0; round < rounds; round++) {
            double[][] aug = new double[n][2 * p];
            for (int i = 0; i < n; i++) {
                System.arraycopy(x[i], 0, aug[i], 0, p);
            }
            for (int j = 0; j < p; j++) {
                int[] perm = new int[n];
                for (int i = 0; i < n; i++) {
                    perm[i] = i;
                }
                for (int i = n - 1; i > 0; i--) {
                    int a = r.nextInt(i + 1);
                    int t = perm[i];
                    perm[i] = perm[a];
                    perm[a] = t;
                }
                for (int i = 0; i < n; i++) {
                    aug[i][p + j] = x[perm[i]][j];
                }
            }
            double[] imp = permutationImportance(aug, y, 60, seed + round);
            double maxShadow = 0;
            for (int j = p; j < 2 * p; j++) {
                maxShadow = Math.max(maxShadow, imp[j]);
            }
            for (int j = 0; j < p; j++) {
                if (imp[j] > maxShadow) {
                    hits[j] += 1.0 / rounds;
                }
            }
        }
        return hits;
    }

    // ------------------------------------------------------------------ neural selectors

    private static Tensor matrix(double[][] x) {
        int n = x.length;
        int p = x[0].length;
        float[] v = new float[n * p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                v[i * p + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(n, p), v);
    }

    private static Tensor column(int[] y) {
        float[] v = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            v[i] = y[i];
        }
        return Tensor.of(Shape.of(y.length, 1), v);
    }

    private static Var bce(Var logits, Tensor y) {
        return logits.softplus().sub(logits.mul(y)).mean();
    }

    private static double[] stochasticGates(double[][] x, int[] y, double lambda, int steps, long seed) {
        int p = x[0].length;
        int h = 32;
        SplittableRandom r = new SplittableRandom(seed);
        Var xs = Var.of(matrix(x));
        Tensor yt = column(y);
        float sigma = 0.5f;
        float[] m0 = new float[p];
        Arrays.fill(m0, 0.5f);
        Var mu = Var.param(Tensor.of(Shape.of(1, p), m0));
        float b1 = (float) Math.sqrt(6.0 / (p + h));
        Var w1 = Var.param(Tensor.uniform(Shape.of(p, h), -b1, b1, r));
        Var c1 = Var.param(Tensor.zeros(Shape.of(1, h)));
        float b2 = (float) Math.sqrt(6.0 / (h + 1));
        Var w2 = Var.param(Tensor.uniform(Shape.of(h, 1), -b2, b2, r));
        Var c2 = Var.param(Tensor.zeros(Shape.of(1, 1)));
        List<Var> params = List.of(mu, w1, c1, w2, c2);
        Optimizer opt = Optimizer.adam(0.02f);
        for (int s = 0; s < steps; s++) {
            Var eps = Var.of(Tensor.normal(Shape.of(1, p), 0, sigma, r));
            Var gate = mu.add(eps).clamp(0, 1);
            Var logits = xs.mul(gate).matmul(w1).add(c1).relu().matmul(w2).add(c2);
            Var reg = mu.div(sigma * (float) Math.sqrt(2)).erf().add(1f).mul(0.5f).sum().mul((float) lambda);
            Var loss = bce(logits, yt).add(reg);
            loss.backward();
            opt.step(params);
        }
        float[] m = mu.value().toArray();
        double[] out = new double[p];
        for (int j = 0; j < p; j++) {
            out[j] = Math.min(1, Math.max(0, m[j]));
        }
        return out;
    }

    private static double[] lassoNet(double[][] x, int[] y, double hierarchy, long seed) {
        int n = x.length;
        int p = x[0].length;
        int h = 32;
        SplittableRandom r = new SplittableRandom(seed);
        Var xs = Var.of(matrix(x));
        Tensor yt = column(y);
        Var theta = Var.param(Tensor.zeros(Shape.of(p, 1)));
        float b1 = (float) Math.sqrt(6.0 / (p + h));
        Var w1 = Var.param(Tensor.uniform(Shape.of(p, h), -b1, b1, r));
        Var c1 = Var.param(Tensor.zeros(Shape.of(1, h)));
        float b2 = (float) Math.sqrt(6.0 / (h + 1));
        Var w2 = Var.param(Tensor.uniform(Shape.of(h, 1), -b2, b2, r));
        Var c2 = Var.param(Tensor.zeros(Shape.of(1, 1)));
        List<Var> params = List.of(theta, w1, c1, w2, c2);
        double lr = 0.05;
        double[] exit = new double[p];
        Arrays.fill(exit, -1);
        // dense warm-up, then an increasing lambda path with the hierarchical prox after each step
        double lambda = 0;
        double lambdaStart = 1e-3;
        for (int stage = 0; stage < 40; stage++) {
            int steps = stage == 0 ? 200 : 25;
            for (int s = 0; s < steps; s++) {
                Var logits = xs.matmul(theta).add(xs.matmul(w1).add(c1).relu().matmul(w2)).add(c2);
                bce(logits, yt).backward();
                for (Var v : params) {
                    Tensor g = v.grad();
                    if (g != null) {
                        v.assign(v.value().sub(g.mul((float) lr)));
                    }
                    v.zeroGrad();
                }
                if (lambda > 0) {
                    hierProx(theta, w1, lambda * lr, hierarchy);
                }
            }
            float[] th = theta.value().toArray();
            for (int j = 0; j < p; j++) {
                if (exit[j] < 0 && th[j] == 0 && lambda > 0) {
                    exit[j] = lambda;
                }
            }
            lambda = stage == 0 ? lambdaStart : lambda * 1.25;
        }
        for (int j = 0; j < p; j++) {
            if (exit[j] < 0) {
                exit[j] = lambda * 2; // never left: most important
            }
        }
        return exit;
    }

    /** The hierarchical proximal operator of LassoNet, applied per feature in place. */
    private static void hierProx(Var theta, Var w1, double lam, double m) {
        float[] th = theta.value().toArray();
        float[] w = w1.value().toArray();
        int p = th.length;
        int h = w.length / p;
        for (int j = 0; j < p; j++) {
            double v = th[j];
            double[] u = new double[h];
            for (int k = 0; k < h; k++) {
                u[k] = Math.abs(w[j * h + k]);
            }
            double[] sorted = u.clone();
            Arrays.sort(sorted);
            // descending
            for (int a = 0, b = h - 1; a < b; a++, b--) {
                double t = sorted[a];
                sorted[a] = sorted[b];
                sorted[b] = t;
            }
            double wm = 0;
            double cum = 0;
            for (int mm = 0; mm <= h; mm++) {
                if (mm > 0) {
                    cum += sorted[mm - 1];
                }
                double cand = m / (1 + mm * m * m) * Math.max(0, Math.abs(v) + m * cum - lam);
                double upper = mm == 0 ? Double.POSITIVE_INFINITY : sorted[mm - 1];
                double lower = mm == h ? 0 : sorted[mm];
                if (cand <= upper && cand >= lower) {
                    wm = cand;
                    break;
                }
            }
            th[j] = (float) (Math.signum(v) * wm / m);
            for (int k = 0; k < h; k++) {
                float s = Math.signum(w[j * h + k]);
                w[j * h + k] = (float) (s * Math.min(wm, u[k]));
            }
        }
        theta.assign(Tensor.of(theta.value().shape(), th));
        w1.assign(Tensor.of(w1.value().shape(), w));
    }
}
