package io.github.zzuegg.brainiac.experimental.graph;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Hyperbolic embeddings of hierarchies (Nickel and Kiela, "Poincaré Embeddings for Learning
 * Hierarchical Representations", NeurIPS 2017, <a href="https://arxiv.org/abs/1705.08039">
 * arXiv:1705.08039</a>; "Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic
 * Geometry", ICML 2018, <a href="https://arxiv.org/abs/1806.03417">arXiv:1806.03417</a>). Trees
 * grow exponentially with depth and so does the volume of hyperbolic space, so a hierarchy embeds
 * in very few hyperbolic dimensions with low distortion.
 * <ul>
 *   <li>{@link Geometry#POINCARE}: points in the open unit ball,
 *       {@code d(u, v) = arcosh(1 + 2 |u - v|^2 / ((1 - |u|^2)(1 - |v|^2)))}, trained by Riemannian
 *       SGD: the Euclidean gradient times {@code (1 - |u|^2)^2 / 4}, then projected back inside the
 *       ball;</li>
 *   <li>{@link Geometry#LORENTZ}: points on the hyperboloid {@code <x, x>_L = -1},
 *       {@code d = arcosh(-<x, y>_L)}, trained by Riemannian SGD with the exact exponential map,
 *       which is numerically stabler near the boundary;</li>
 *   <li>{@link Geometry#EUCLIDEAN}: the same loss with Euclidean distances, the baseline.</li>
 * </ul>
 * The loss is the papers' softmax ranking over distances: for an edge {@code (u, v)} and sampled
 * non-neighbours {@code v'}, {@code -log(e^(-d(u,v)) / sum_(w in {v} + negatives) e^(-d(u,w)))}.
 * The first epochs are a burn-in at a tenth of the learning rate, as in the paper.
 */
public final class HyperbolicEmbedding {

    /** The embedding space. */
    public enum Geometry {
        /** The Poincaré ball. */
        POINCARE,
        /** The hyperboloid (Lorentz) model. */
        LORENTZ,
        /** Flat Euclidean space. */
        EUCLIDEAN
    }

    private static final double EPS = 1e-5;

    private final Geometry geometry;
    private final int dim;
    private final double[][] x;   // Lorentz points carry dim + 1 coordinates

    /**
     * Creates embeddings near the origin.
     *
     * @param geometry the space
     * @param nodes the number of nodes
     * @param dim the dimension
     * @param seed the initialisation seed
     */
    public HyperbolicEmbedding(Geometry geometry, int nodes, int dim, long seed) {
        this.geometry = geometry;
        this.dim = dim;
        SplittableRandom r = new SplittableRandom(seed);
        x = new double[nodes][geometry == Geometry.LORENTZ ? dim + 1 : dim];
        for (double[] p : x) {
            int off = geometry == Geometry.LORENTZ ? 1 : 0;
            for (int k = 0; k < dim; k++) {
                p[off + k] = (r.nextDouble() * 2 - 1) * 1e-3;
            }
            if (geometry == Geometry.LORENTZ) {
                p[0] = lorentzTime(p);
            }
        }
    }

    private static double lorentzTime(double[] p) {
        double s = 1;
        for (int k = 1; k < p.length; k++) {
            s += p[k] * p[k];
        }
        return Math.sqrt(s);
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int k = 0; k < a.length; k++) {
            s += a[k] * b[k];
        }
        return s;
    }

    private static double minkowski(double[] a, double[] b) {
        return dot(a, b) - 2 * a[0] * b[0];
    }

    private static double arcosh(double z) {
        return Math.log(z + Math.sqrt(Math.max(z * z - 1, 0)));
    }

    /**
     * @param u a node
     * @param v another node
     * @return their distance in the embedding space
     */
    public double distance(int u, int v) {
        return distance(x[u], x[v]);
    }

    private double distance(double[] a, double[] b) {
        return switch (geometry) {
            case EUCLIDEAN -> {
                double s = 0;
                for (int k = 0; k < a.length; k++) {
                    s += (a[k] - b[k]) * (a[k] - b[k]);
                }
                yield Math.sqrt(s);
            }
            case POINCARE -> {
                double sq = 0;
                for (int k = 0; k < a.length; k++) {
                    sq += (a[k] - b[k]) * (a[k] - b[k]);
                }
                double alpha = 1 - dot(a, a);
                double beta = 1 - dot(b, b);
                yield arcosh(1 + 2 * sq / Math.max(alpha * beta, EPS));
            }
            case LORENTZ -> arcosh(Math.max(-minkowski(a, b), 1 + 1e-12));
        };
    }

    /**
     * The gradient of {@code d(a, b)} with respect to {@code a}, Euclidean in the ambient
     * coordinates.
     */
    private double[] distanceGradient(double[] a, double[] b) {
        double[] g = new double[a.length];
        switch (geometry) {
            case EUCLIDEAN -> {
                double d = Math.max(distance(a, b), 1e-9);
                for (int k = 0; k < a.length; k++) {
                    g[k] = (a[k] - b[k]) / d;
                }
            }
            case POINCARE -> {
                double sqA = dot(a, a);
                double sqB = dot(b, b);
                double alpha = Math.max(1 - sqA, EPS);
                double beta = Math.max(1 - sqB, EPS);
                double sq = 0;
                for (int k = 0; k < a.length; k++) {
                    sq += (a[k] - b[k]) * (a[k] - b[k]);
                }
                double gamma = 1 + 2 * sq / (alpha * beta);
                double scale = 4 / (beta * Math.sqrt(Math.max(gamma * gamma - 1, 1e-12)));
                double coef = (sqB - 2 * dot(a, b) + 1) / (alpha * alpha);
                for (int k = 0; k < a.length; k++) {
                    g[k] = scale * (coef * a[k] - b[k] / alpha);
                }
            }
            case LORENTZ -> {
                double m = -minkowski(a, b);
                double s = -1 / Math.sqrt(Math.max(m * m - 1, 1e-12));
                // d/da of -<a, b>_L is (b0, -b1, ..., -bd)
                g[0] = s * b[0];
                for (int k = 1; k < a.length; k++) {
                    g[k] = -s * b[k];
                }
            }
        }
        return g;
    }

    /** Riemannian SGD step on one point with a Euclidean gradient. */
    private void step(double[] p, double[] grad, double lr) {
        switch (geometry) {
            case EUCLIDEAN -> {
                for (int k = 0; k < p.length; k++) {
                    p[k] -= lr * grad[k];
                }
            }
            case POINCARE -> {
                double f = Math.pow(1 - dot(p, p), 2) / 4;
                for (int k = 0; k < p.length; k++) {
                    p[k] -= lr * f * grad[k];
                }
                double norm = Math.sqrt(dot(p, p));
                if (norm >= 1 - EPS) {
                    for (int k = 0; k < p.length; k++) {
                        p[k] *= (1 - EPS) / norm;
                    }
                }
            }
            case LORENTZ -> {
                // h = g_L^-1 grad (flip the time coordinate), project onto the tangent space at p
                double[] h = grad.clone();
                h[0] = -h[0];
                double inner = minkowski(p, h);
                for (int k = 0; k < p.length; k++) {
                    h[k] = -lr * (h[k] + inner * p[k]);
                }
                double norm = Math.sqrt(Math.max(minkowski(h, h), 0));
                if (norm < 1e-12) {
                    return;
                }
                norm = Math.min(norm, 1);
                double ch = Math.cosh(norm);
                double sh = Math.sinh(norm) / Math.sqrt(Math.max(minkowski(h, h), 1e-24));
                for (int k = 0; k < p.length; k++) {
                    p[k] = ch * p[k] + sh * h[k];
                }
                p[0] = lorentzTime(p);
            }
        }
    }

    /**
     * Trains on undirected edges with the softmax ranking loss.
     *
     * @param edges pairs {@code {u, v}}
     * @param epochs passes over the edges
     * @param negatives sampled non-neighbours per edge
     * @param lr the learning rate
     * @param burnIn epochs at {@code lr / 10} first
     * @param seed the sampling seed
     * @return the final epoch's mean loss
     */
    public double train(List<int[]> edges, int epochs, int negatives, double lr, int burnIn, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Set<Long> known = new HashSet<>();
        for (int[] e : edges) {
            known.add(key(e[0], e[1]));
            known.add(key(e[1], e[0]));
        }
        int n = x.length;
        int[] order = new int[edges.size()];
        double last = 0;
        for (int epoch = 0; epoch < epochs + burnIn; epoch++) {
            double rate = epoch < burnIn ? lr / 10 : lr;
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            for (int i = order.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            double total = 0;
            for (int idx : order) {
                int[] e = edges.get(idx);
                int u = r.nextBoolean() ? e[0] : e[1];
                int v = u == e[0] ? e[1] : e[0];
                int[] cand = new int[negatives + 1];
                cand[0] = v;
                for (int k = 1; k <= negatives; k++) {
                    int w;
                    int tries = 0;
                    do {
                        w = r.nextInt(n);
                    } while ((w == u || known.contains(key(u, w))) && ++tries < 20);
                    cand[k] = w;
                }
                double[] d = new double[cand.length];
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < cand.length; k++) {
                    d[k] = distance(x[u], x[cand[k]]);
                    max = Math.max(max, -d[k]);
                }
                double z = 0;
                for (int k = 0; k < cand.length; k++) {
                    z += Math.exp(-d[k] - max);
                }
                total += d[0] + max + Math.log(z);
                double[] gu = new double[x[u].length];
                double[][] gw = new double[cand.length][];
                for (int k = 0; k < cand.length; k++) {
                    double coef = (k == 0 ? 1 : 0) - Math.exp(-d[k] - max) / z;
                    double[] a = distanceGradient(x[u], x[cand[k]]);
                    double[] b = distanceGradient(x[cand[k]], x[u]);
                    for (int c = 0; c < gu.length; c++) {
                        gu[c] += coef * a[c];
                        b[c] *= coef;
                    }
                    gw[k] = b;
                }
                step(x[u], gu, rate);
                for (int k = 0; k < cand.length; k++) {
                    step(x[cand[k]], gw[k], rate);
                }
            }
            last = total / Math.max(1, order.length);
        }
        return last;
    }

    private static long key(int a, int b) {
        return (long) a << 32 | (b & 0xffffffffL);
    }

    /**
     * @param u a node
     * @return its distance from the origin (the hierarchy level a hyperbolic embedding encodes)
     */
    public double norm(int u) {
        return switch (geometry) {
            case EUCLIDEAN, POINCARE -> {
                double[] o = new double[dim];
                yield distance(o, x[u]);
            }
            case LORENTZ -> arcosh(Math.max(x[u][0], 1));
        };
    }

    /** @return the geometry */
    public Geometry geometry() {
        return geometry;
    }

    /** @return the number of nodes */
    public int nodes() {
        return x.length;
    }
}
