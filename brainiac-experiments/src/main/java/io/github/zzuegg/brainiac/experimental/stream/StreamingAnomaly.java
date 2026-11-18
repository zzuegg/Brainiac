package io.github.zzuegg.brainiac.experimental.stream;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Streaming anomaly detectors, used prequentially: {@link Detector#process} scores a point
 * (higher is more anomalous) and then learns from it, with bounded memory and no retraining.
 *
 * <ul>
 *   <li>{@link #halfSpaceTrees}: half-space trees (Tan, Ting and Liu, IJCAI 2011): random
 *       full trees that halve a random dimension of a randomly perturbed work space; each node
 *       counts the mass of a reference window and of the latest window, the windows swap every
 *       {@code ψ} points, and the score is {@code -Σ r_node 2^depth} at the deepest node whose
 *       reference mass is still large enough.</li>
 *   <li>{@link #rrcf}: robust random cut forests (Guha, Mishra, Roy and Schrijvers, ICML 2016):
 *       random cut trees over a sliding window, maintained exactly under insertion and deletion
 *       (cuts are drawn proportionally to the bounding box's side lengths), scored by collusive
 *       displacement, the largest ratio of sibling to subtree size along the point's path.</li>
 *   <li>{@link #loda}: LODA (Pevný, Machine Learning 2016): one-dimensional histograms on sparse
 *       random projections ({@code sqrt(d)} non-zeros), with exponentially fading counts; the
 *       score is the mean negative log density.</li>
 *   <li>{@link #onlineIsolationForest}: Online-iForest (Leveni, Weigert Cassales, Pfahringer,
 *       Bifet and Boracchi, May 2025, <a href="https://arxiv.org/abs/2505.09593">arXiv:2505.09593</a>):
 *       multi-resolution histogram trees over a sliding window of {@code ω} points; a leaf at depth
 *       {@code k} splits (random dimension, uniform value in its support, children initialised
 *       from points sampled in the support) when its height reaches {@code η 2^k} and
 *       {@code k < log2(ω/η)}; points leaving the window decrement heights and merge children
 *       whose parent falls below {@code η 2^k}; the score is {@code 2^(-E[depth] / log2(ω/η))}.</li>
 * </ul>
 */
public final class StreamingAnomaly {

    private StreamingAnomaly() {
    }

    /** A streaming detector. */
    @FunctionalInterface
    public interface Detector {

        /**
         * Scores a point, then learns from it.
         *
         * @param x the point
         * @return its anomaly score before learning, higher is more anomalous
         */
        double process(double[] x);
    }

    // ------------------------------------------------------------------ half-space trees

    /**
     * Half-space trees.
     *
     * @param dims the input dimension
     * @param trees the number of trees, for example 25
     * @param depth the tree depth, for example 15 (clipped to 20)
     * @param window the window {@code ψ}, for example 250
     * @param lo the lower bound of the data's range per dimension (from a warm-up sample)
     * @param hi the upper bound per dimension
     * @param seed the seed
     * @return the detector
     */
    public static Detector halfSpaceTrees(int dims, int trees, int depth, int window, double[] lo, double[] hi,
            long seed) {
        int h = Math.min(depth, 20);
        SplittableRandom r = new SplittableRandom(seed);
        int nodes = (1 << (h + 1)) - 1;
        int[][] dim = new int[trees][nodes];
        double[][] split = new double[trees][nodes];
        for (int t = 0; t < trees; t++) {
            double[] mn = new double[dims];
            double[] mx = new double[dims];
            for (int q = 0; q < dims; q++) {
                double s = lo[q] + r.nextDouble() * (hi[q] - lo[q]);
                double range = 2 * Math.max(s - lo[q], hi[q] - s) + 1e-9;
                mn[q] = s - range;
                mx[q] = s + range;
            }
            build(dim[t], split[t], 0, 0, h, mn, mx, r);
        }
        double[][] ref = new double[trees][nodes];
        double[][] latest = new double[trees][nodes];
        int[] seen = new int[1];
        double sizeLimit = 0.1 * window;
        return x -> {
            double score = 0;
            boolean ready = seen[0] >= window;
            for (int t = 0; t < trees; t++) {
                int node = 0;
                int k = 0;
                boolean scored = false;
                while (true) {
                    if (ready && !scored && (k == h || ref[t][node] < sizeLimit)) {
                        score += ref[t][node] * Math.pow(2, k);
                        scored = true;
                    }
                    latest[t][node]++;
                    if (k == h) {
                        break;
                    }
                    node = x[dim[t][node]] < split[t][node] ? 2 * node + 1 : 2 * node + 2;
                    k++;
                }
            }
            seen[0]++;
            if (seen[0] % window == 0) {
                for (int t = 0; t < trees; t++) {
                    System.arraycopy(latest[t], 0, ref[t], 0, nodes);
                    Arrays.fill(latest[t], 0);
                }
            }
            return ready ? -score / trees : 0;
        };
    }

    private static void build(int[] dim, double[] split, int node, int k, int h, double[] mn, double[] mx,
            SplittableRandom r) {
        if (k == h) {
            return;
        }
        int q = r.nextInt(mn.length);
        double mid = (mn[q] + mx[q]) / 2;
        dim[node] = q;
        split[node] = mid;
        double[] lmx = mx.clone();
        lmx[q] = mid;
        double[] rmn = mn.clone();
        rmn[q] = mid;
        build(dim, split, 2 * node + 1, k + 1, h, mn, lmx, r);
        build(dim, split, 2 * node + 2, k + 1, h, rmn, mx, r);
    }

    // ------------------------------------------------------------------ RRCF

    /**
     * Robust random cut forest.
     *
     * @param trees the number of trees, for example 40
     * @param window the points each tree keeps, for example 256
     * @param seed the seed
     * @return the detector
     */
    public static Detector rrcf(int trees, int window, long seed) {
        CutTree[] forest = new CutTree[trees];
        SplittableRandom r = new SplittableRandom(seed);
        for (int t = 0; t < trees; t++) {
            forest[t] = new CutTree(r.split());
        }
        ArrayDeque<Long> ids = new ArrayDeque<>();
        long[] next = new long[1];
        return x -> {
            long id = next[0]++;
            double s = 0;
            for (CutTree t : forest) {
                s += t.insert(x, id);
            }
            ids.addLast(id);
            if (ids.size() > window) {
                long old = ids.removeFirst();
                for (CutTree t : forest) {
                    t.delete(old);
                }
            }
            return s / trees;
        };
    }

    private static final class CutNode {
        CutNode left;
        CutNode right;
        CutNode parent;
        int dim;
        double cut;
        double[] lo;
        double[] hi;
        int size;
        double[] point;

        boolean leaf() {
            return point != null;
        }
    }

    /** One random cut tree with exact insertion and deletion. */
    private static final class CutTree {
        private final SplittableRandom random;
        private final Map<Long, CutNode> leaves = new HashMap<>();
        private CutNode root;

        CutTree(SplittableRandom random) {
            this.random = random;
        }

        /** Inserts a point and returns its collusive displacement. */
        double insert(double[] x, long id) {
            CutNode leaf = new CutNode();
            leaf.point = x.clone();
            leaf.lo = x.clone();
            leaf.hi = x.clone();
            leaf.size = 1;
            if (root == null) {
                root = leaf;
            } else {
                CutNode existing = findDuplicate(root, x);
                if (existing != null) {
                    leaves.put(id, existing);
                    for (CutNode n = existing; n != null; n = n.parent) {
                        n.size++;
                    }
                    return codisp(existing);
                }
                root = insertAt(root, leaf);
                root.parent = null;
            }
            leaves.put(id, leaf);
            return codisp(leaf);
        }

        private static CutNode findDuplicate(CutNode node, double[] x) {
            CutNode n = node;
            while (!n.leaf()) {
                n = x[n.dim] <= n.cut ? n.left : n.right;
            }
            return Arrays.equals(n.point, x) ? n : null;
        }

        private CutNode insertAt(CutNode node, CutNode leaf) {
            double[] p = leaf.point;
            int d = p.length;
            double[] lo = new double[d];
            double[] hi = new double[d];
            double total = 0;
            for (int j = 0; j < d; j++) {
                lo[j] = Math.min(node.lo[j], p[j]);
                hi[j] = Math.max(node.hi[j], p[j]);
                total += hi[j] - lo[j];
            }
            double u = random.nextDouble() * total;
            int dim = 0;
            double cut = lo[0];
            for (int j = 0; j < d; j++) {
                double span = hi[j] - lo[j];
                if (u <= span || j == d - 1) {
                    dim = j;
                    cut = lo[j] + Math.min(u, span);
                    break;
                }
                u -= span;
            }
            boolean below = cut < node.lo[dim];
            boolean above = cut >= node.hi[dim];
            if (below || above || node.leaf()) {
                CutNode branch = new CutNode();
                branch.dim = dim;
                branch.cut = cut;
                if (below || (!above && p[dim] <= cut)) {
                    branch.left = leaf;
                    branch.right = node;
                } else {
                    branch.left = node;
                    branch.right = leaf;
                }
                branch.left.parent = branch;
                branch.right.parent = branch;
                branch.lo = lo;
                branch.hi = hi;
                branch.size = node.size + leaf.size;
                return branch;
            }
            if (p[node.dim] <= node.cut) {
                node.left = insertAt(node.left, leaf);
                node.left.parent = node;
            } else {
                node.right = insertAt(node.right, leaf);
                node.right.parent = node;
            }
            node.lo = lo;
            node.hi = hi;
            node.size += leaf.size;
            return node;
        }

        void delete(long id) {
            CutNode leaf = leaves.remove(id);
            if (leaf == null) {
                return;
            }
            if (leaf.size > 1) {
                for (CutNode n = leaf; n != null; n = n.parent) {
                    n.size--;
                }
                return;
            }
            CutNode parent = leaf.parent;
            if (parent == null) {
                root = null;
                return;
            }
            CutNode sibling = parent.left == leaf ? parent.right : parent.left;
            CutNode grand = parent.parent;
            sibling.parent = grand;
            if (grand == null) {
                root = sibling;
                return;
            }
            if (grand.left == parent) {
                grand.left = sibling;
            } else {
                grand.right = sibling;
            }
            for (CutNode n = grand; n != null; n = n.parent) {
                n.size = n.left.size + n.right.size;
                for (int j = 0; j < n.lo.length; j++) {
                    n.lo[j] = Math.min(n.left.lo[j], n.right.lo[j]);
                    n.hi[j] = Math.max(n.left.hi[j], n.right.hi[j]);
                }
            }
        }

        private static double codisp(CutNode leaf) {
            double best = 0;
            for (CutNode n = leaf; n.parent != null; n = n.parent) {
                CutNode sib = n.parent.left == n ? n.parent.right : n.parent.left;
                best = Math.max(best, sib.size / (double) n.size);
            }
            return best;
        }
    }

    // ------------------------------------------------------------------ LODA

    /**
     * LODA with fading histograms.
     *
     * @param dims the input dimension
     * @param projections the number of projections, for example 100
     * @param bins histogram bins per projection
     * @param warmup points used to fix the bin ranges (scored 0)
     * @param halfLife the fading half-life of the counts, in points
     * @param seed the seed
     * @return the detector
     */
    public static Detector loda(int dims, int projections, int bins, int warmup, double halfLife, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] w = new double[projections][dims];
        int nz = Math.max(1, (int) Math.round(Math.sqrt(dims)));
        for (double[] row : w) {
            int[] order = new int[dims];
            for (int j = 0; j < dims; j++) {
                order[j] = j;
            }
            for (int j = 0; j < nz; j++) {
                int k = j + r.nextInt(dims - j);
                int tmp = order[j];
                order[j] = order[k];
                order[k] = tmp;
                row[order[j]] = r.nextGaussian();
            }
        }
        double decay = Math.pow(0.5, 1 / halfLife);
        double[][] counts = new double[projections][bins];
        double[] lo = new double[projections];
        double[] width = new double[projections];
        double[] total = new double[1];
        java.util.List<double[]> buffer = new java.util.ArrayList<>();
        return x -> {
            double[] z = new double[projections];
            for (int k = 0; k < projections; k++) {
                for (int j = 0; j < dims; j++) {
                    z[k] += w[k][j] * x[j];
                }
            }
            if (buffer.size() < warmup) {
                buffer.add(z);
                if (buffer.size() == warmup) {
                    for (int k = 0; k < projections; k++) {
                        double mn = Double.POSITIVE_INFINITY;
                        double mx = Double.NEGATIVE_INFINITY;
                        for (double[] b : buffer) {
                            mn = Math.min(mn, b[k]);
                            mx = Math.max(mx, b[k]);
                        }
                        double pad = 0.1 * (mx - mn) + 1e-9;
                        lo[k] = mn - pad;
                        width[k] = (mx - mn + 2 * pad) / bins;
                    }
                    for (double[] b : buffer) {
                        add(b, counts, lo, width, total, decay);
                    }
                }
                return 0;
            }
            double s = 0;
            for (int k = 0; k < projections; k++) {
                int bin = (int) Math.floor((z[k] - lo[k]) / width[k]);
                double c = bin >= 0 && bin < bins ? counts[k][bin] : 0;
                double density = (c + 0.1) / ((total[0] + 0.1 * bins) * width[k]);
                s -= Math.log(density);
            }
            add(z, counts, lo, width, total, decay);
            return s / projections;
        };
    }

    private static void add(double[] z, double[][] counts, double[] lo, double[] width, double[] total, double decay) {
        total[0] = total[0] * decay + 1;
        for (int k = 0; k < z.length; k++) {
            double[] c = counts[k];
            for (int b = 0; b < c.length; b++) {
                c[b] *= decay;
            }
            int bin = (int) Math.floor((z[k] - lo[k]) / width[k]);
            if (bin >= 0 && bin < c.length) {
                c[bin]++;
            }
        }
    }

    // ------------------------------------------------------------------ Online-iForest

    /**
     * Online isolation forest.
     *
     * @param trees the number of trees {@code τ}, for example 32
     * @param window the sliding window {@code ω}, for example 512
     * @param eta the points needed to split a bin {@code η}, for example 16
     * @param seed the seed
     * @return the detector
     */
    public static Detector onlineIsolationForest(int trees, int window, int eta, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        HistogramTree[] forest = new HistogramTree[trees];
        int maxDepth = (int) Math.floor(Math.log((double) window / eta) / Math.log(2));
        for (int t = 0; t < trees; t++) {
            forest[t] = new HistogramTree(eta, maxDepth, r.split());
        }
        double c = Math.max(1, Math.log((double) window / eta) / Math.log(2));
        ArrayDeque<double[]> buffer = new ArrayDeque<>();
        return x -> {
            double depth = 0;
            for (HistogramTree t : forest) {
                depth += t.depth(x);
            }
            double score = Math.pow(2, -(depth / trees) / c);
            buffer.addLast(x.clone());
            for (HistogramTree t : forest) {
                t.learn(x);
            }
            if (buffer.size() > window) {
                double[] old = buffer.removeFirst();
                for (HistogramTree t : forest) {
                    t.forget(old);
                }
            }
            return score;
        };
    }

    private static final class HistogramNode {
        double height;
        double[] lo;
        double[] hi;
        int dim;
        double value;
        HistogramNode left;
        HistogramNode right;
    }

    /** One Online-iTree: a multi-resolution histogram. */
    private static final class HistogramTree {
        private final int eta;
        private final int maxDepth;
        private final SplittableRandom random;
        private HistogramNode root;

        HistogramTree(int eta, int maxDepth, SplittableRandom random) {
            this.eta = eta;
            this.maxDepth = maxDepth;
            this.random = random;
        }

        int depth(double[] x) {
            if (root == null) {
                return 0;
            }
            int k = 0;
            HistogramNode n = root;
            while (n.left != null) {
                n = x[n.dim] < n.value ? n.left : n.right;
                k++;
            }
            return k;
        }

        void learn(double[] x) {
            if (root == null) {
                root = new HistogramNode();
                root.lo = x.clone();
                root.hi = x.clone();
            }
            HistogramNode n = root;
            int k = 0;
            while (true) {
                n.height++;
                for (int j = 0; j < x.length; j++) {
                    n.lo[j] = Math.min(n.lo[j], x[j]);
                    n.hi[j] = Math.max(n.hi[j], x[j]);
                }
                if (n.left == null) {
                    if (n.height >= eta * Math.pow(2, k) && k < maxDepth) {
                        // the new children are initialised from sampled points; the new point
                        // is then passed down to the child that holds it
                        split(n, k);
                    } else {
                        return;
                    }
                }
                n = x[n.dim] < n.value ? n.left : n.right;
                k++;
            }
        }

        private void split(HistogramNode n, int k) {
            int d = n.lo.length;
            int q = random.nextInt(d);
            n.dim = q;
            n.value = n.lo[q] + random.nextDouble() * (n.hi[q] - n.lo[q]);
            HistogramNode l = new HistogramNode();
            HistogramNode rr = new HistogramNode();
            l.lo = new double[d];
            l.hi = new double[d];
            rr.lo = new double[d];
            rr.hi = new double[d];
            Arrays.fill(l.lo, Double.POSITIVE_INFINITY);
            Arrays.fill(l.hi, Double.NEGATIVE_INFINITY);
            Arrays.fill(rr.lo, Double.POSITIVE_INFINITY);
            Arrays.fill(rr.hi, Double.NEGATIVE_INFINITY);
            int samples = (int) Math.min(n.height, eta * Math.pow(2, k));
            double[] s = new double[d];
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < d; j++) {
                    s[j] = n.lo[j] + random.nextDouble() * (n.hi[j] - n.lo[j]);
                }
                HistogramNode t = s[q] < n.value ? l : rr;
                t.height++;
                for (int j = 0; j < d; j++) {
                    t.lo[j] = Math.min(t.lo[j], s[j]);
                    t.hi[j] = Math.max(t.hi[j], s[j]);
                }
            }
            for (HistogramNode c : new HistogramNode[] {l, rr}) {
                if (c.height == 0) {
                    c.lo = n.lo.clone();
                    c.hi = n.hi.clone();
                }
            }
            n.left = l;
            n.right = rr;
        }

        void forget(double[] x) {
            HistogramNode n = root;
            int k = 0;
            while (n != null) {
                n.height = Math.max(0, n.height - 1);
                if (n.left == null) {
                    return;
                }
                if (n.height < eta * Math.pow(2, k)) {
                    for (int j = 0; j < n.lo.length; j++) {
                        n.lo[j] = Math.min(n.left.lo[j], n.right.lo[j]);
                        n.hi[j] = Math.max(n.left.hi[j], n.right.hi[j]);
                    }
                    n.left = null;
                    n.right = null;
                    return;
                }
                n = x[n.dim] < n.value ? n.left : n.right;
                k++;
            }
        }
    }
}
