package io.github.zzuegg.brainiac.experimental.stream;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Stream clustering: one pass, bounded memory, and an offline step that turns the online summary
 * into clusters on demand ({@link Clusterer#recluster()}).
 *
 * <ul>
 *   <li>{@link #sequentialKMeans}: MacQueen's online k-means, the nearest centre moves by
 *       {@code (x - c) / n_c}; capping {@code n_c} turns the {@code 1/n} step into a constant
 *       one that tracks drift.</li>
 *   <li>{@link #miniBatchKMeans}: Sculley's mini-batch k-means (WWW 2010): assign a batch to the
 *       current centres, then move each centre with its own per-centre rate {@code 1/v_c}.</li>
 *   <li>{@link #birch}: BIRCH (Zhang, Ramakrishnan and Livny, SIGMOD 1996): a height-balanced
 *       CF tree of clustering features {@code (N, LS, SS)}; a point joins the nearest leaf entry
 *       if its radius stays below the threshold {@code T}, nodes split at the branching factor,
 *       and the threshold grows (rebuilding the tree from its leaf entries) when the leaves
 *       exceed their budget. The offline step is weighted k-means on the leaf centroids.</li>
 *   <li>{@link #cluStream}: CluStream (Aggarwal, Han, Wang and Yu, VLDB 2003): {@code q}
 *       micro-clusters with temporal extensions {@code (CF2t, CF1t)}; a point joins the nearest
 *       one inside its maximal boundary (a factor times its RMS deviation), otherwise it starts
 *       a new one and the least recently relevant micro-cluster is deleted or the two closest
 *       are merged. Pyramidal snapshots (orders {@code 2^i}, {@code 2^l + 1} per order) let the
 *       offline weighted k-means cluster only a recent horizon by subtracting the snapshot.</li>
 *   <li>{@link #denStream}: DenStream (Cao, Ester, Qian and Zhou, SDM 2006): potential and
 *       outlier micro-clusters whose weights fade as {@code 2^(-λ Δt)}; a point joins the nearest
 *       potential (else outlier) micro-cluster whose radius stays below {@code ε}, outliers that
 *       reach weight {@code βμ} are promoted, and periodic pruning drops faded ones. The offline
 *       step is DBSCAN on the potential micro-clusters, so cluster count and shape are free and
 *       noise is labelled {@code -1}.</li>
 * </ul>
 */
public final class StreamClustering {

    private StreamClustering() {
    }

    /** A stream clusterer. */
    public interface Clusterer {

        /**
         * Absorbs one point.
         *
         * @param x the point
         */
        void learn(double[] x);

        /** Runs the offline step on the current summary (cheap; call before {@link #assign}). */
        void recluster();

        /**
         * The cluster of a point under the last {@link #recluster()}.
         *
         * @param x the point
         * @return a cluster id, or {@code -1} for noise
         */
        int assign(double[] x);

        /** @return the number of summaries kept (centres, CF entries or micro-clusters) */
        int summaries();
    }

    // ------------------------------------------------------------------ helpers

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int j = 0; j < a.length; j++) {
            double d = a[j] - b[j];
            s += d * d;
        }
        return s;
    }

    private static int nearest(double[][] centres, double[] x) {
        int best = 0;
        double bd = Double.POSITIVE_INFINITY;
        for (int i = 0; i < centres.length; i++) {
            double d = dist2(centres[i], x);
            if (d < bd) {
                bd = d;
                best = i;
            }
        }
        return best;
    }

    /**
     * Weighted k-means (k-means++ seeding, Lloyd iterations) on summary centroids.
     *
     * @param points the centroids
     * @param weights their weights
     * @param k the number of clusters
     * @param seed the seed
     * @return the centres
     */
    public static double[][] weightedKMeans(double[][] points, double[] weights, int k, long seed) {
        int n = points.length;
        if (n == 0) {
            return new double[0][];
        }
        k = Math.min(k, n);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] c = new double[k][];
        double total = Arrays.stream(weights).sum();
        double u = r.nextDouble() * total;
        int first = 0;
        for (double acc = 0; first < n - 1; first++) {
            acc += weights[first];
            if (acc >= u) {
                break;
            }
        }
        c[0] = points[first].clone();
        double[] d = new double[n];
        for (int j = 1; j < k; j++) {
            double s = 0;
            for (int i = 0; i < n; i++) {
                d[i] = Double.POSITIVE_INFINITY;
                for (int m = 0; m < j; m++) {
                    d[i] = Math.min(d[i], dist2(points[i], c[m]));
                }
                d[i] *= weights[i];
                s += d[i];
            }
            double v = r.nextDouble() * s;
            int pick = 0;
            for (double acc = 0; pick < n - 1; pick++) {
                acc += d[pick];
                if (acc >= v) {
                    break;
                }
            }
            c[j] = points[pick].clone();
        }
        int dim = points[0].length;
        for (int it = 0; it < 20; it++) {
            double[][] sum = new double[k][dim];
            double[] w = new double[k];
            for (int i = 0; i < n; i++) {
                int a = nearest(c, points[i]);
                w[a] += weights[i];
                for (int j = 0; j < dim; j++) {
                    sum[a][j] += weights[i] * points[i][j];
                }
            }
            for (int a = 0; a < k; a++) {
                if (w[a] > 0) {
                    for (int j = 0; j < dim; j++) {
                        c[a][j] = sum[a][j] / w[a];
                    }
                }
            }
        }
        return c;
    }

    // ------------------------------------------------------------------ k-means variants

    /**
     * Sequential (MacQueen) k-means.
     *
     * @param k the number of centres
     * @param maxCount the cap on each centre's count (the smallest step is {@code 1/maxCount});
     *     {@code Integer.MAX_VALUE} for the classic {@code 1/n}
     * @return the clusterer
     */
    public static Clusterer sequentialKMeans(int k, int maxCount) {
        return new KMeansStream(k, 1, maxCount);
    }

    /**
     * Mini-batch k-means.
     *
     * @param k the number of centres
     * @param batch the batch size
     * @param maxCount the cap on each centre's count
     * @return the clusterer
     */
    public static Clusterer miniBatchKMeans(int k, int batch, int maxCount) {
        return new KMeansStream(k, batch, maxCount);
    }

    private static final class KMeansStream implements Clusterer {
        private final int k;
        private final int batch;
        private final int maxCount;
        private final List<double[]> centres = new ArrayList<>();
        private final List<double[]> pending = new ArrayList<>();
        private int[] counts;

        KMeansStream(int k, int batch, int maxCount) {
            this.k = k;
            this.batch = batch;
            this.maxCount = maxCount;
            this.counts = new int[k];
        }

        @Override
        public void learn(double[] x) {
            if (centres.size() < k) {
                for (double[] c : centres) {
                    if (dist2(c, x) == 0) {
                        return;
                    }
                }
                centres.add(x.clone());
                counts[centres.size() - 1] = 1;
                return;
            }
            pending.add(x.clone());
            if (pending.size() < batch) {
                return;
            }
            double[][] c = centres.toArray(new double[0][]);
            int[] assign = new int[pending.size()];
            for (int i = 0; i < assign.length; i++) {
                assign[i] = nearest(c, pending.get(i));
            }
            for (int i = 0; i < assign.length; i++) {
                int a = assign[i];
                counts[a] = Math.min(maxCount, counts[a] + 1);
                double eta = 1.0 / counts[a];
                double[] cc = centres.get(a);
                double[] p = pending.get(i);
                for (int j = 0; j < cc.length; j++) {
                    cc[j] += eta * (p[j] - cc[j]);
                }
            }
            pending.clear();
        }

        @Override
        public void recluster() {
        }

        @Override
        public int assign(double[] x) {
            return centres.isEmpty() ? 0 : nearest(centres.toArray(new double[0][]), x);
        }

        @Override
        public int summaries() {
            return centres.size();
        }
    }

    // ------------------------------------------------------------------ BIRCH

    /**
     * BIRCH.
     *
     * @param threshold the initial radius threshold {@code T}
     * @param branching the maximum entries per node
     * @param maxLeafEntries the leaf entry budget before the threshold grows
     * @param k the clusters of the offline step
     * @return the clusterer
     */
    public static Clusterer birch(double threshold, int branching, int maxLeafEntries, int k) {
        return new Birch(threshold, branching, maxLeafEntries, k);
    }

    /** A clustering feature (N, LS, SS). */
    private static final class Cf {
        double n;
        double[] ls;
        double ss;

        Cf(double n, double[] ls, double ss) {
            this.n = n;
            this.ls = ls;
            this.ss = ss;
        }

        static Cf point(double[] x) {
            double s = 0;
            for (double v : x) {
                s += v * v;
            }
            return new Cf(1, x.clone(), s);
        }

        Cf copy() {
            return new Cf(n, ls.clone(), ss);
        }

        void add(Cf o) {
            n += o.n;
            for (int j = 0; j < ls.length; j++) {
                ls[j] += o.ls[j];
            }
            ss += o.ss;
        }

        double[] centroid() {
            double[] c = new double[ls.length];
            for (int j = 0; j < c.length; j++) {
                c[j] = ls[j] / n;
            }
            return c;
        }

        double radiusWith(Cf o) {
            double nn = n + o.n;
            double s = ss + o.ss;
            double c2 = 0;
            for (int j = 0; j < ls.length; j++) {
                double m = (ls[j] + o.ls[j]) / nn;
                c2 += m * m;
            }
            return Math.sqrt(Math.max(0, s / nn - c2));
        }
    }

    private static final class BirchNode {
        final boolean leaf;
        final List<Cf> entries = new ArrayList<>();
        final List<BirchNode> children = new ArrayList<>();

        BirchNode(boolean leaf) {
            this.leaf = leaf;
        }

        Cf summary() {
            Cf s = entries.get(0).copy();
            for (int i = 1; i < entries.size(); i++) {
                s.add(entries.get(i));
            }
            return s;
        }
    }

    private static final class Birch implements Clusterer {
        private double threshold;
        private final int branching;
        private final int maxLeafEntries;
        private final int k;
        private BirchNode root = new BirchNode(true);
        private int leafEntries;
        private double[][] centres = new double[0][];

        Birch(double threshold, int branching, int maxLeafEntries, int k) {
            this.threshold = threshold;
            this.branching = branching;
            this.maxLeafEntries = maxLeafEntries;
            this.k = k;
        }

        @Override
        public void learn(double[] x) {
            insertRoot(Cf.point(x));
            if (leafEntries > maxLeafEntries) {
                List<Cf> leaves = new ArrayList<>();
                collect(root, leaves);
                while (leafEntries > maxLeafEntries / 2) {
                    threshold *= 1.5;
                    root = new BirchNode(true);
                    leafEntries = 0;
                    for (Cf cf : leaves) {
                        insertRoot(cf);
                    }
                    leaves.clear();
                    collect(root, leaves);
                }
            }
        }

        private void insertRoot(Cf cf) {
            BirchNode split = insert(root, cf);
            if (split != null) {
                BirchNode r = new BirchNode(false);
                r.children.add(root);
                r.entries.add(root.summary());
                r.children.add(split);
                r.entries.add(split.summary());
                root = r;
            }
        }

        private static int closest(List<Cf> entries, double[] c) {
            int best = 0;
            double bd = Double.POSITIVE_INFINITY;
            for (int i = 0; i < entries.size(); i++) {
                double d = dist2(entries.get(i).centroid(), c);
                if (d < bd) {
                    bd = d;
                    best = i;
                }
            }
            return best;
        }

        /** Inserts and returns a new sibling if the node split. */
        private BirchNode insert(BirchNode node, Cf cf) {
            double[] c = cf.centroid();
            if (node.leaf) {
                if (!node.entries.isEmpty()) {
                    int i = closest(node.entries, c);
                    if (node.entries.get(i).radiusWith(cf) <= threshold) {
                        node.entries.get(i).add(cf);
                        return null;
                    }
                }
                node.entries.add(cf.copy());
                leafEntries++;
                return node.entries.size() > branching ? split(node) : null;
            }
            int i = closest(node.entries, c);
            BirchNode child = node.children.get(i);
            BirchNode sib = insert(child, cf);
            node.entries.set(i, child.summary());
            if (sib != null) {
                node.children.add(sib);
                node.entries.add(sib.summary());
                if (node.entries.size() > branching) {
                    return split(node);
                }
            }
            return null;
        }

        /** Splits a node around its farthest pair of entries. */
        private BirchNode split(BirchNode node) {
            int n = node.entries.size();
            int a = 0;
            int b = 1;
            double far = -1;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double d = dist2(node.entries.get(i).centroid(), node.entries.get(j).centroid());
                    if (d > far) {
                        far = d;
                        a = i;
                        b = j;
                    }
                }
            }
            double[] ca = node.entries.get(a).centroid();
            double[] cb = node.entries.get(b).centroid();
            List<Cf> entries = new ArrayList<>(node.entries);
            List<BirchNode> children = new ArrayList<>(node.children);
            node.entries.clear();
            node.children.clear();
            BirchNode other = new BirchNode(node.leaf);
            for (int i = 0; i < n; i++) {
                double[] ci = entries.get(i).centroid();
                boolean toA = i == a || (i != b && dist2(ci, ca) <= dist2(ci, cb));
                BirchNode target = toA ? node : other;
                target.entries.add(entries.get(i));
                if (!node.leaf) {
                    target.children.add(children.get(i));
                }
            }
            return other;
        }

        private static void collect(BirchNode node, List<Cf> out) {
            if (node.leaf) {
                out.addAll(node.entries);
            } else {
                for (BirchNode c : node.children) {
                    collect(c, out);
                }
            }
        }

        @Override
        public void recluster() {
            List<Cf> leaves = new ArrayList<>();
            collect(root, leaves);
            double[][] pts = new double[leaves.size()][];
            double[] w = new double[leaves.size()];
            for (int i = 0; i < pts.length; i++) {
                pts[i] = leaves.get(i).centroid();
                w[i] = leaves.get(i).n;
            }
            centres = weightedKMeans(pts, w, k, 7);
        }

        @Override
        public int assign(double[] x) {
            return centres.length == 0 ? 0 : nearest(centres, x);
        }

        @Override
        public int summaries() {
            return leafEntries;
        }
    }

    // ------------------------------------------------------------------ CluStream

    /**
     * CluStream.
     *
     * @param q the number of micro-clusters
     * @param k the clusters of the offline step
     * @param boundary the maximal boundary factor {@code t} (2 in the paper)
     * @param horizon the time horizon of the offline step in points (0 for everything)
     * @return the clusterer
     */
    public static Clusterer cluStream(int q, int k, double boundary, int horizon) {
        return new CluStream(q, k, boundary, horizon);
    }

    private static final class MicroCluster {
        double n;
        double[] ls;
        double[] ss;
        double lst;
        double sst;
        final Set<Integer> ids = new HashSet<>();

        MicroCluster(double[] x, double time, int id) {
            n = 1;
            ls = x.clone();
            ss = new double[x.length];
            for (int j = 0; j < x.length; j++) {
                ss[j] = x[j] * x[j];
            }
            lst = time;
            sst = time * time;
            ids.add(id);
        }

        MicroCluster(MicroCluster o) {
            n = o.n;
            ls = o.ls.clone();
            ss = o.ss.clone();
            lst = o.lst;
            sst = o.sst;
            ids.addAll(o.ids);
        }

        void absorb(double[] x, double time) {
            n++;
            for (int j = 0; j < x.length; j++) {
                ls[j] += x[j];
                ss[j] += x[j] * x[j];
            }
            lst += time;
            sst += time * time;
        }

        void merge(MicroCluster o, double sign) {
            n += sign * o.n;
            for (int j = 0; j < ls.length; j++) {
                ls[j] += sign * o.ls[j];
                ss[j] += sign * o.ss[j];
            }
            lst += sign * o.lst;
            sst += sign * o.sst;
        }

        double[] centre() {
            double[] c = new double[ls.length];
            for (int j = 0; j < c.length; j++) {
                c[j] = ls[j] / n;
            }
            return c;
        }

        double rms() {
            double s = 0;
            for (int j = 0; j < ls.length; j++) {
                double m = ls[j] / n;
                s += ss[j] / n - m * m;
            }
            return Math.sqrt(Math.max(0, s / ls.length));
        }

        /** Mean plus one standard deviation of the time stamps: the "relevance stamp". */
        double relevance() {
            double m = lst / n;
            return m + Math.sqrt(Math.max(0, sst / n - m * m));
        }
    }

    private static final class CluStream implements Clusterer {
        private final int q;
        private final int k;
        private final double boundary;
        private final int horizon;
        private final List<MicroCluster> clusters = new ArrayList<>();
        /** Snapshots by order: lists of (time, micro-clusters). */
        private final List<List<Object[]>> snapshots = new ArrayList<>();
        private int time;
        private int nextId;
        private double[][] centres = new double[0][];

        CluStream(int q, int k, double boundary, int horizon) {
            this.q = q;
            this.k = k;
            this.boundary = boundary;
            this.horizon = horizon;
        }

        @Override
        public void learn(double[] x) {
            time++;
            if (clusters.size() < q) {
                clusters.add(new MicroCluster(x, time, nextId++));
            } else {
                int best = -1;
                double bd = Double.POSITIVE_INFINITY;
                for (int i = 0; i < clusters.size(); i++) {
                    double d = dist2(clusters.get(i).centre(), x);
                    if (d < bd) {
                        bd = d;
                        best = i;
                    }
                }
                MicroCluster m = clusters.get(best);
                double radius;
                if (m.n > 1) {
                    radius = boundary * m.rms();
                } else {
                    radius = Double.POSITIVE_INFINITY;
                    for (int i = 0; i < clusters.size(); i++) {
                        if (i != best) {
                            radius = Math.min(radius, Math.sqrt(dist2(clusters.get(i).centre(), m.centre())));
                        }
                    }
                }
                if (Math.sqrt(bd) <= radius) {
                    m.absorb(x, time);
                } else {
                    makeRoom();
                    clusters.add(new MicroCluster(x, time, nextId++));
                }
            }
            snapshot();
        }

        private void makeRoom() {
            double cutoff = time - Math.max(horizon, 4 * q);
            int oldest = -1;
            double least = Double.POSITIVE_INFINITY;
            for (int i = 0; i < clusters.size(); i++) {
                double r = clusters.get(i).relevance();
                if (r < least) {
                    least = r;
                    oldest = i;
                }
            }
            if (least < cutoff) {
                clusters.remove(oldest);
                return;
            }
            int a = 0;
            int b = 1;
            double bd = Double.POSITIVE_INFINITY;
            for (int i = 0; i < clusters.size(); i++) {
                for (int j = i + 1; j < clusters.size(); j++) {
                    double d = dist2(clusters.get(i).centre(), clusters.get(j).centre());
                    if (d < bd) {
                        bd = d;
                        a = i;
                        b = j;
                    }
                }
            }
            MicroCluster ma = clusters.get(a);
            MicroCluster mb = clusters.get(b);
            ma.merge(mb, 1);
            ma.ids.addAll(mb.ids);
            clusters.remove(b);
        }

        private void snapshot() {
            int order = Integer.numberOfTrailingZeros(time);
            while (snapshots.size() <= order) {
                snapshots.add(new ArrayList<>());
            }
            List<MicroCluster> copy = new ArrayList<>();
            for (MicroCluster m : clusters) {
                copy.add(new MicroCluster(m));
            }
            List<Object[]> level = snapshots.get(order);
            level.add(new Object[] {time, copy});
            if (level.size() > 5) {
                level.remove(0);
            }
        }

        @SuppressWarnings("unchecked")
        private List<MicroCluster> windowed() {
            if (horizon <= 0 || time <= horizon) {
                return clusters;
            }
            int target = time - horizon;
            List<MicroCluster> best = null;
            int bestGap = Integer.MAX_VALUE;
            for (List<Object[]> level : snapshots) {
                for (Object[] s : level) {
                    int gap = Math.abs((Integer) s[0] - target);
                    if (gap < bestGap) {
                        bestGap = gap;
                        best = (List<MicroCluster>) s[1];
                    }
                }
            }
            if (best == null) {
                return clusters;
            }
            List<MicroCluster> out = new ArrayList<>();
            for (MicroCluster m : clusters) {
                MicroCluster d = new MicroCluster(m);
                for (MicroCluster old : best) {
                    if (m.ids.containsAll(old.ids)) {
                        d.merge(old, -1);
                    }
                }
                if (d.n >= 1) {
                    out.add(d);
                }
            }
            return out.isEmpty() ? clusters : out;
        }

        @Override
        public void recluster() {
            List<MicroCluster> mc = windowed();
            double[][] pts = new double[mc.size()][];
            double[] w = new double[mc.size()];
            for (int i = 0; i < pts.length; i++) {
                pts[i] = mc.get(i).centre();
                w[i] = mc.get(i).n;
            }
            centres = weightedKMeans(pts, w, k, 7);
        }

        @Override
        public int assign(double[] x) {
            return centres.length == 0 ? 0 : nearest(centres, x);
        }

        @Override
        public int summaries() {
            return clusters.size();
        }
    }

    // ------------------------------------------------------------------ DenStream

    /**
     * DenStream.
     *
     * @param eps the micro-cluster radius bound {@code ε}
     * @param mu the core weight {@code μ}
     * @param beta the potential factor {@code β} (outliers are promoted at weight {@code βμ})
     * @param lambda the fading rate {@code λ} per point
     * @return the clusterer
     */
    public static Clusterer denStream(double eps, double mu, double beta, double lambda) {
        return new DenStream(eps, mu, beta, lambda, 2 * eps);
    }

    /**
     * DenStream with an explicit offline connection radius (the offline DBSCAN joins two
     * potential micro-clusters whose centres are within {@code offlineEps}; the default is
     * {@code 2ε}, which fragments thin manifolds whose micro-clusters sit further apart).
     *
     * @param eps the micro-cluster radius bound {@code ε}
     * @param mu the core weight {@code μ}
     * @param beta the potential factor {@code β}
     * @param lambda the fading rate {@code λ} per point
     * @param offlineEps the offline connection radius
     * @return the clusterer
     */
    public static Clusterer denStream(double eps, double mu, double beta, double lambda, double offlineEps) {
        return new DenStream(eps, mu, beta, lambda, offlineEps);
    }

    private static final class Dmc {
        double w;
        double[] cf1;
        double[] cf2;
        double last;
        final double created;

        Dmc(double[] x, double t) {
            w = 1;
            cf1 = x.clone();
            cf2 = new double[x.length];
            for (int j = 0; j < x.length; j++) {
                cf2[j] = x[j] * x[j];
            }
            last = t;
            created = t;
        }

        void fade(double t, double lambda) {
            double f = Math.pow(2, -lambda * (t - last));
            w *= f;
            for (int j = 0; j < cf1.length; j++) {
                cf1[j] *= f;
                cf2[j] *= f;
            }
            last = t;
        }

        double[] centre() {
            double[] c = new double[cf1.length];
            for (int j = 0; j < c.length; j++) {
                c[j] = cf1[j] / w;
            }
            return c;
        }

        double radiusWith(double[] x) {
            double nw = w + 1;
            double s = 0;
            for (int j = 0; j < cf1.length; j++) {
                double m = (cf1[j] + x[j]) / nw;
                s += (cf2[j] + x[j] * x[j]) / nw - m * m;
            }
            return Math.sqrt(Math.max(0, s));
        }

        double radius() {
            double s = 0;
            for (int j = 0; j < cf1.length; j++) {
                double m = cf1[j] / w;
                s += cf2[j] / w - m * m;
            }
            return Math.sqrt(Math.max(0, s));
        }

        void absorb(double[] x) {
            w++;
            for (int j = 0; j < cf1.length; j++) {
                cf1[j] += x[j];
                cf2[j] += x[j] * x[j];
            }
        }
    }

    private static final class DenStream implements Clusterer {
        private final double eps;
        private final double mu;
        private final double beta;
        private final double lambda;
        private final int period;
        private final double offlineEps;
        private final List<Dmc> potential = new ArrayList<>();
        private final List<Dmc> outliers = new ArrayList<>();
        private double time;
        private double[][] centres = new double[0][];
        private int[] labels = new int[0];
        private double[] radii = new double[0];

        DenStream(double eps, double mu, double beta, double lambda, double offlineEps) {
            this.eps = eps;
            this.offlineEps = offlineEps;
            this.mu = mu;
            this.beta = beta;
            this.lambda = lambda;
            double bm = beta * mu;
            this.period = bm > 1 ? Math.max(1, (int) Math.ceil(1 / lambda * Math.log(bm / (bm - 1)) / Math.log(2)))
                    : 100;
        }

        private static int nearestTo(List<Dmc> list, double[] x) {
            int best = -1;
            double bd = Double.POSITIVE_INFINITY;
            for (int i = 0; i < list.size(); i++) {
                double d = dist2(list.get(i).centre(), x);
                if (d < bd) {
                    bd = d;
                    best = i;
                }
            }
            return best;
        }

        @Override
        public void learn(double[] x) {
            time++;
            int p = nearestTo(potential, x);
            if (p >= 0) {
                Dmc m = potential.get(p);
                m.fade(time, lambda);
                if (m.radiusWith(x) <= eps) {
                    m.absorb(x);
                    prune();
                    return;
                }
            }
            int o = nearestTo(outliers, x);
            if (o >= 0) {
                Dmc m = outliers.get(o);
                m.fade(time, lambda);
                if (m.radiusWith(x) <= eps) {
                    m.absorb(x);
                    if (m.w > beta * mu) {
                        outliers.remove(o);
                        potential.add(m);
                    }
                    prune();
                    return;
                }
            }
            outliers.add(new Dmc(x, time));
            prune();
        }

        private void prune() {
            if (((long) time) % period != 0) {
                return;
            }
            potential.removeIf(m -> {
                m.fade(time, lambda);
                return m.w < beta * mu;
            });
            double denom = Math.pow(2, -lambda * period) - 1;
            outliers.removeIf(m -> {
                m.fade(time, lambda);
                double xi = (Math.pow(2, -lambda * (time - m.created + period)) - 1) / denom;
                return m.w < xi;
            });
        }

        @Override
        public void recluster() {
            int n = potential.size();
            centres = new double[n][];
            radii = new double[n];
            labels = new int[n];
            Arrays.fill(labels, -1);
            for (int i = 0; i < n; i++) {
                potential.get(i).fade(time, lambda);
                centres[i] = potential.get(i).centre();
                radii[i] = potential.get(i).radius();
            }
            int next = 0;
            for (int i = 0; i < n; i++) {
                if (labels[i] >= 0) {
                    continue;
                }
                labels[i] = next;
                java.util.ArrayDeque<Integer> queue = new java.util.ArrayDeque<>();
                queue.add(i);
                while (!queue.isEmpty()) {
                    int a = queue.poll();
                    for (int b = 0; b < n; b++) {
                        if (labels[b] < 0 && Math.sqrt(dist2(centres[a], centres[b])) <= offlineEps) {
                            labels[b] = next;
                            queue.add(b);
                        }
                    }
                }
                next++;
            }
        }

        @Override
        public int assign(double[] x) {
            if (centres.length == 0) {
                return -1;
            }
            int i = nearest(centres, x);
            double d = Math.sqrt(dist2(centres[i], x));
            return d <= Math.max(eps, 2 * radii[i]) ? labels[i] : -1;
        }

        @Override
        public int summaries() {
            return potential.size() + outliers.size();
        }
    }
}
