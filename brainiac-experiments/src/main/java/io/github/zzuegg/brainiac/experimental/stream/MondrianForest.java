package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Mondrian forests for online classification and regression (Lakshminarayanan, Roy and Teh,
 * NeurIPS 2014, <a href="https://arxiv.org/abs/1406.2673">arXiv:1406.2673</a>, and AISTATS 2016,
 * <a href="https://arxiv.org/abs/1506.03805">arXiv:1506.03805</a>).
 *
 * <p>A Mondrian tree splits a box at a time drawn from an exponential race whose rate is the box's
 * linear dimension {@code Σ (hi - lo)}, along a dimension chosen proportionally to its side, so a
 * new point can be added <em>exactly in distribution</em>: if the point lies outside a node's box,
 * a split separating it may be inserted above the node ({@code τ_parent + Exp(Σ e) < τ_node},
 * with {@code e} the box's extension towards the point); otherwise the box grows and the point
 * descends. Leaves pause while pure (classification) or small (regression) and resample their
 * block when that changes; the lifetime {@code λ} bounds the depth.
 *
 * <p>Predictions integrate over where the test point would branch off the path: at each node the
 * point separates with probability {@code 1 - exp(-Δτ η(x))}, {@code η(x)} its distance outside
 * the box. Classification uses the hierarchical (interpolated Kneser–Ney approximation of the)
 * normalised stable process prior, {@code G_j = (c_j - d tab_j + d Σtab_j G_parent) / Σ c_j} with
 * discount {@code d = exp(-γ Δτ)}; regression uses a Gaussian prior along the tree whose variance
 * grows from 0 at the root to the target's variance as {@code 1 - exp(-γ τ)}, each node's posterior
 * combining its parent's with its own data mean (a simplified version of the paper's Gaussian
 * belief propagation, which also passes messages upwards). Far from the data the prediction falls back towards
 * the root, so uncertainty widens in gaps of the input space.
 */
public final class MondrianForest implements OnlineClassifier {

    private final List<Tree> trees = new ArrayList<>();
    private final int dims;
    private final int classes;
    private final boolean regression;
    private final double lifetime;
    private final double gamma;
    private final int minRegressionLeaf;
    private final List<double[]> xs = new ArrayList<>();
    private final List<Integer> ys = new ArrayList<>();
    private final List<Double> values = new ArrayList<>();
    private double ySum;
    private double ySq;

    private MondrianForest(int trees, int dims, int classes, boolean regression, double lifetime, double gamma,
            int minRegressionLeaf, long seed) {
        this.dims = dims;
        this.classes = classes;
        this.regression = regression;
        this.lifetime = lifetime;
        this.gamma = gamma;
        this.minRegressionLeaf = minRegressionLeaf;
        SplittableRandom r = new SplittableRandom(seed);
        for (int t = 0; t < trees; t++) {
            this.trees.add(new Tree(r.split()));
        }
    }

    /**
     * A classification forest.
     *
     * @param trees the number of trees, for example 10
     * @param dims the input dimension
     * @param classes the number of classes
     * @param lifetime the lifetime {@code λ} ({@code Double.POSITIVE_INFINITY} for none)
     * @param gamma the discount rate {@code γ}, about {@code 10 / (typical feature range)}
     * @param seed the seed
     * @return the forest
     */
    public static MondrianForest classifier(int trees, int dims, int classes, double lifetime, double gamma,
            long seed) {
        return new MondrianForest(trees, dims, classes, false, lifetime, gamma, 0, seed);
    }

    /**
     * A regression forest.
     *
     * @param trees the number of trees
     * @param dims the input dimension
     * @param lifetime the lifetime {@code λ}
     * @param varianceRate the rate {@code γ} at which the prior variance approaches the target's
     *     variance along split time
     * @param minLeaf leaves with fewer points do not split
     * @param seed the seed
     * @return the forest
     */
    public static MondrianForest regressor(int trees, int dims, double lifetime, double varianceRate, int minLeaf,
            long seed) {
        return new MondrianForest(trees, dims, 1, true, lifetime, varianceRate, minLeaf, seed);
    }

    // ------------------------------------------------------------------ the tree

    private final class Node {
        Node parent;
        Node left;
        Node right;
        int dim;
        double cut;
        double tau;
        double[] lo;
        double[] hi;
        double[] counts;
        double n;
        double sum;
        double sq;
        List<Integer> points;

        boolean leaf() {
            return left == null;
        }

        double parentTau() {
            return parent == null ? 0 : parent.tau;
        }
    }

    private final class Tree {
        private final SplittableRandom random;
        private Node root;

        Tree(SplittableRandom random) {
            this.random = random;
        }

        private double exp(double rate) {
            return rate <= 0 ? Double.POSITIVE_INFINITY : -Math.log(1 - random.nextDouble()) / rate;
        }

        private boolean paused(List<Integer> pts) {
            if (regression) {
                return pts.size() < minRegressionLeaf;
            }
            int first = ys.get(pts.get(0));
            for (int p : pts) {
                if (ys.get(p) != first) {
                    return false;
                }
            }
            return true;
        }

        /** SampleMondrianBlock: builds a subtree on a set of points. */
        private Node sample(Node parent, List<Integer> pts) {
            Node node = new Node();
            node.parent = parent;
            node.lo = new double[dims];
            node.hi = new double[dims];
            Arrays.fill(node.lo, Double.POSITIVE_INFINITY);
            Arrays.fill(node.hi, Double.NEGATIVE_INFINITY);
            for (int p : pts) {
                double[] x = xs.get(p);
                for (int j = 0; j < dims; j++) {
                    node.lo[j] = Math.min(node.lo[j], x[j]);
                    node.hi[j] = Math.max(node.hi[j], x[j]);
                }
            }
            double rate = 0;
            for (int j = 0; j < dims; j++) {
                rate += node.hi[j] - node.lo[j];
            }
            double start = parent == null ? 0 : parent.tau;
            double e = exp(rate);
            if (paused(pts) || start + e >= lifetime) {
                node.tau = lifetime;
                node.points = new ArrayList<>(pts);
                return node;
            }
            node.tau = start + e;
            double u = random.nextDouble() * rate;
            int d = 0;
            for (; d < dims - 1; d++) {
                double side = node.hi[d] - node.lo[d];
                if (u < side) {
                    break;
                }
                u -= side;
            }
            node.dim = d;
            node.cut = node.lo[d] + random.nextDouble() * (node.hi[d] - node.lo[d]);
            List<Integer> l = new ArrayList<>();
            List<Integer> r = new ArrayList<>();
            for (int p : pts) {
                (xs.get(p)[d] <= node.cut ? l : r).add(p);
            }
            if (l.isEmpty() || r.isEmpty()) {
                node.tau = lifetime;
                node.points = new ArrayList<>(pts);
                return node;
            }
            node.left = sample(node, l);
            node.right = sample(node, r);
            return node;
        }

        void add(int p) {
            if (root == null) {
                root = sample(null, List.of(p));
            } else {
                root = extend(root, p);
                root.parent = null;
            }
            recount(root);
        }

        /** ExtendMondrianBlock; returns the node that now stands where {@code node} stood. */
        private Node extend(Node node, int p) {
            double[] x = xs.get(p);
            double[] el = new double[dims];
            double[] eu = new double[dims];
            double rate = 0;
            for (int j = 0; j < dims; j++) {
                el[j] = Math.max(node.lo[j] - x[j], 0);
                eu[j] = Math.max(x[j] - node.hi[j], 0);
                rate += el[j] + eu[j];
            }
            double e = exp(rate);
            if (node.parentTau() + e < node.tau) {
                double u = random.nextDouble() * rate;
                int d = 0;
                for (; d < dims - 1; d++) {
                    double side = el[d] + eu[d];
                    if (u < side) {
                        break;
                    }
                    u -= side;
                }
                Node split = new Node();
                split.parent = node.parent;
                split.tau = node.parentTau() + e;
                split.dim = d;
                split.cut = x[d] > node.hi[d] ? node.hi[d] + random.nextDouble() * (x[d] - node.hi[d])
                        : x[d] + random.nextDouble() * (node.lo[d] - x[d]);
                split.lo = node.lo.clone();
                split.hi = node.hi.clone();
                for (int j = 0; j < dims; j++) {
                    split.lo[j] = Math.min(split.lo[j], x[j]);
                    split.hi[j] = Math.max(split.hi[j], x[j]);
                }
                Node leaf = sample(split, List.of(p));
                node.parent = split;
                if (x[d] <= split.cut) {
                    split.left = leaf;
                    split.right = node;
                } else {
                    split.left = node;
                    split.right = leaf;
                }
                return split;
            }
            for (int j = 0; j < dims; j++) {
                node.lo[j] = Math.min(node.lo[j], x[j]);
                node.hi[j] = Math.max(node.hi[j], x[j]);
            }
            if (node.leaf()) {
                node.points.add(p);
                if (!paused(node.points)) {
                    Node fresh = sample(node.parent, node.points);
                    return fresh;
                }
                return node;
            }
            if (x[node.dim] <= node.cut) {
                node.left = extend(node.left, p);
                node.left.parent = node;
            } else {
                node.right = extend(node.right, p);
                node.right.parent = node;
            }
            return node;
        }

        /** Recomputes counts (tables for classification, sufficient statistics for regression). */
        private void recount(Node node) {
            if (node.leaf()) {
                node.counts = new double[classes];
                node.n = 0;
                node.sum = 0;
                node.sq = 0;
                for (int p : node.points) {
                    if (regression) {
                        double v = values.get(p);
                        node.sum += v;
                        node.sq += v * v;
                    } else {
                        node.counts[ys.get(p)]++;
                    }
                    node.n++;
                }
                return;
            }
            recount(node.left);
            recount(node.right);
            node.counts = new double[classes];
            for (int c = 0; c < classes; c++) {
                node.counts[c] = Math.min(node.left.counts[c], 1) + Math.min(node.right.counts[c], 1);
            }
            node.n = node.left.n + node.right.n;
            node.sum = node.left.sum + node.right.sum;
            node.sq = node.left.sq + node.right.sq;
        }

        private double[] posterior(double[] counts, double[] prior, double discount) {
            double total = 0;
            double tables = 0;
            for (int c = 0; c < classes; c++) {
                total += counts[c];
                tables += Math.min(counts[c], 1);
            }
            if (total == 0) {
                return prior.clone();
            }
            double[] g = new double[classes];
            for (int c = 0; c < classes; c++) {
                g[c] = (counts[c] - discount * Math.min(counts[c], 1) + discount * tables * prior[c]) / total;
            }
            return g;
        }

        double[] classify(double[] x) {
            double[] base = new double[classes];
            Arrays.fill(base, 1.0 / classes);
            double[] out = new double[classes];
            if (root == null) {
                return base;
            }
            double notSeparated = 1;
            double[] prior = base;
            Node node = root;
            while (true) {
                double delta = span(node);
                double eta = distanceOutside(node, x);
                double ps = eta == 0 ? 0 : 1 - Math.exp(-delta * eta);
                if (ps > 0) {
                    double meanDelta = Double.isInfinite(delta) ? 1 / eta
                            : 1 / eta - delta / Math.expm1(Math.min(700, eta * delta));
                    double[] tab = new double[classes];
                    for (int c = 0; c < classes; c++) {
                        tab[c] = Math.min(node.counts[c], 1);
                    }
                    double[] g = posterior(tab, prior, Math.exp(-gamma * meanDelta));
                    for (int c = 0; c < classes; c++) {
                        out[c] += notSeparated * ps * g[c];
                    }
                }
                double[] g = posterior(node.counts, prior,
                        Math.exp(-gamma * (Double.isInfinite(delta) ? 1e9 : delta)));
                if (node.leaf()) {
                    for (int c = 0; c < classes; c++) {
                        out[c] += notSeparated * (1 - ps) * g[c];
                    }
                    return out;
                }
                notSeparated *= 1 - ps;
                prior = g;
                node = x[node.dim] <= node.cut ? node.left : node.right;
            }
        }

        /** {mean, variance} of the regression prediction. */
        double[] regress(double[] x, double noise, double rootMean, double rootVar) {
            if (root == null) {
                return new double[] {rootMean, rootVar + noise};
            }
            double notSeparated = 1;
            double pm = rootMean;
            double pv = 0;
            double m1 = 0;
            double m2 = 0;
            Node node = root;
            while (true) {
                double delta = span(node);
                double eta = distanceOutside(node, x);
                double ps = eta == 0 ? 0 : 1 - Math.exp(-delta * eta);
                if (ps > 0) {
                    double meanDelta = Double.isInfinite(delta) ? 1 / eta
                            : 1 / eta - delta / Math.expm1(Math.min(700, eta * delta));
                    double t0 = node.parentTau();
                    double v = pv + rootVar * (growth(t0 + meanDelta) - growth(t0)) + noise;
                    m1 += notSeparated * ps * pm;
                    m2 += notSeparated * ps * (v + pm * pm);
                }
                double prior = pv + rootVar * (growth(node.parentTau() + delta) - growth(node.parentTau())) + 1e-12;
                double mean = pm;
                double var = prior;
                if (node.n > 0) {
                    double lik = noise / node.n;
                    double ybar = node.sum / node.n;
                    var = 1 / (1 / prior + 1 / lik);
                    mean = var * (pm / prior + ybar / lik);
                }
                if (node.leaf()) {
                    double v = var + noise;
                    m1 += notSeparated * (1 - ps) * mean;
                    m2 += notSeparated * (1 - ps) * (v + mean * mean);
                    return new double[] {m1, Math.max(1e-12, m2 - m1 * m1)};
                }
                notSeparated *= 1 - ps;
                pm = mean;
                pv = var;
                node = x[node.dim] <= node.cut ? node.left : node.right;
            }
        }

        /** The share of the prior variance accumulated by time {@code t}, {@code 1 - exp(-γ t)}. */
        private double growth(double t) {
            return -Math.expm1(-gamma * t);
        }

        /**
         * The node's lifetime {@code τ - τ_parent}; a leaf that never split (infinite lifetime)
         * uses the expected time to its next split, {@code 1 / (linear dimension)}.
         */
        private double span(Node node) {
            if (Double.isFinite(node.tau)) {
                return node.tau - node.parentTau();
            }
            double size = 0;
            for (int j = 0; j < dims; j++) {
                size += node.hi[j] - node.lo[j];
            }
            return 1 / Math.max(size, 1e-6);
        }

        private double distanceOutside(Node node, double[] x) {
            double e = 0;
            for (int j = 0; j < dims; j++) {
                e += Math.max(node.lo[j] - x[j], 0) + Math.max(x[j] - node.hi[j], 0);
            }
            return e;
        }

        int leaves(Node node) {
            return node == null ? 0 : node.leaf() ? 1 : leaves(node.left) + leaves(node.right);
        }
    }

    // ------------------------------------------------------------------ API

    /**
     * Learns a labelled point (classification).
     *
     * @param x the input
     * @param y the class
     */
    public void learn(double[] x, int y) {
        xs.add(x.clone());
        ys.add(y);
        values.add(0.0);
        int p = xs.size() - 1;
        for (Tree t : trees) {
            t.add(p);
        }
    }

    /**
     * Learns a point with a real target (regression).
     *
     * @param x the input
     * @param y the target
     */
    public void learnValue(double[] x, double y) {
        xs.add(x.clone());
        ys.add(0);
        values.add(y);
        ySum += y;
        ySq += y * y;
        int p = xs.size() - 1;
        for (Tree t : trees) {
            t.add(p);
        }
    }

    /**
     * The class probabilities, averaged over the trees.
     *
     * @param x the input
     * @return one probability per class
     */
    public double[] predictProbability(double[] x) {
        double[] p = new double[classes];
        for (Tree t : trees) {
            double[] q = t.classify(x);
            for (int c = 0; c < classes; c++) {
                p[c] += q[c] / trees.size();
            }
        }
        return p;
    }

    /**
     * The regression prediction: the mixture over trees of each tree's Gaussian.
     *
     * @param x the input
     * @return {@code {mean, variance}}
     */
    public double[] predictMeanVariance(double[] x) {
        int n = values.size();
        double mean = n == 0 ? 0 : ySum / n;
        double var = n < 2 ? 1 : Math.max(1e-9, ySq / n - mean * mean);
        double noise = noiseVariance(var);
        double m1 = 0;
        double m2 = 0;
        for (Tree t : trees) {
            double[] mv = t.regress(x, noise, mean, var);
            m1 += mv[0] / trees.size();
            m2 += (mv[1] + mv[0] * mv[0]) / trees.size();
        }
        return new double[] {m1, Math.max(1e-12, m2 - m1 * m1)};
    }

    /** Pooled within-leaf variance of the first tree, a floor of 1% of the total. */
    private double noiseVariance(double total) {
        double[] acc = new double[2];
        pool(trees.get(0).root, acc);
        double v = acc[1] > 0 ? acc[0] / acc[1] : total;
        return Math.max(0.01 * total, v);
    }

    private static void pool(MondrianForest.Node node, double[] acc) {
        if (node == null) {
            return;
        }
        if (node.leaf()) {
            if (node.n > 1) {
                double m = node.sum / node.n;
                acc[0] += node.sq - node.n * m * m;
                acc[1] += node.n - 1;
            }
            return;
        }
        pool(node.left, acc);
        pool(node.right, acc);
    }

    @Override
    public double[] predict(DriftingStreams.Sample sample) {
        return predictProbability(sample.x());
    }

    @Override
    public void learn(DriftingStreams.Sample sample) {
        learn(sample.x(), sample.label());
    }

    /** @return the mean number of leaves per tree */
    public double meanLeaves() {
        double s = 0;
        for (Tree t : trees) {
            s += t.leaves(t.root);
        }
        return s / trees.size();
    }

    @Override
    public long bytes() {
        return 8L * xs.size() * (dims + 1) + (long) (meanLeaves() * trees.size() * 8 * (2 * dims + classes + 6));
    }
}
