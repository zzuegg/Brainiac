package io.github.zzuegg.brainiac.experimental.boost;

import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * LightGBM's three additions to second-order histogram boosting (Ke et al., "LightGBM: A Highly
 * Efficient Gradient Boosting Decision Tree", NeurIPS 2017,
 * <a href="https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree">
 * paper</a>), on top of the same quantile bins and {@link GradientBoostedTrees.Loss} derivatives as
 * core {@link GradientBoostedTrees}:
 * <ul>
 *   <li><em>gradient-based one-side sampling</em> (GOSS): each tree keeps the {@code topRate}
 *       share of rows with the largest gradients and a random {@code otherRate} share of the rest,
 *       whose gradients and curvatures are multiplied by {@code (1 - topRate) / otherRate} so the
 *       split gains stay nearly unbiased;</li>
 *   <li><em>exclusive feature bundling</em> (EFB): features that are rarely away from their most
 *       common bin at the same time (one-hot or count columns) are merged by greedy colouring of
 *       their conflict graph into one histogram column with offset bins; splits are still found per
 *       original feature, recovering its default bin as the bundle total minus its other bins;</li>
 *   <li><em>leaf-wise (best-first) growth</em> under a leaf budget, rather than level by level,
 *       with histogram subtraction (the larger child's histogram is the parent's minus the
 *       smaller child's).</li>
 * </ul>
 * Setting {@code leafWise = false} grows level by level (shallowest leaf first), which is the core
 * booster's order. An optional per-row <em>base margin</em> replaces the constant starting score
 * (LightGBM's {@code init_score}), which {@link PriorBoost} uses to boost on top of another model.
 */
public final class LightBoost {

    /**
     * Training settings.
     *
     * @param trees boosting rounds
     * @param maxLeaves the leaf budget per tree
     * @param maxDepth the depth cap, 0 for none
     * @param learningRate the shrinkage
     * @param lambda L2 regularisation of leaf values
     * @param minChildWeight the least summed curvature per leaf
     * @param minDataInLeaf the fewest (sampled) rows per leaf
     * @param bins thresholds per feature
     * @param topRate GOSS's large-gradient share {@code a}; 0 turns GOSS off
     * @param otherRate GOSS's random share {@code b} of the remaining rows
     * @param bundle whether exclusive features are bundled
     * @param maxConflictRate the share of rows on which bundled features may clash
     * @param leafWise best-first growth (true) or level by level (false)
     * @param seed seeds GOSS sampling
     */
    public record Settings(int trees, int maxLeaves, int maxDepth, double learningRate, double lambda,
            double minChildWeight, int minDataInLeaf, int bins, double topRate, double otherRate, boolean bundle,
            double maxConflictRate, boolean leafWise, long seed) {

        /** @return 100 leaf-wise trees of up to 31 leaves, learning rate 0.1, no GOSS, no bundling */
        public static Settings defaults() {
            return new Settings(100, 31, 0, 0.1, 1, 1e-3, 20, 32, 0, 0, false, 0, true, 1);
        }

        /**
         * @param n rounds
         * @return a copy with that many trees
         */
        public Settings withTrees(int n) {
            return new Settings(n, maxLeaves, maxDepth, learningRate, lambda, minChildWeight, minDataInLeaf, bins,
                    topRate, otherRate, bundle, maxConflictRate, leafWise, seed);
        }

        /**
         * @param leaves the leaf budget
         * @param depth the depth cap, 0 for none
         * @return a copy with that budget and cap
         */
        public Settings withLeaves(int leaves, int depth) {
            return new Settings(trees, leaves, depth, learningRate, lambda, minChildWeight, minDataInLeaf, bins,
                    topRate, otherRate, bundle, maxConflictRate, leafWise, seed);
        }

        /**
         * @param a the kept large-gradient share
         * @param b the sampled share of the rest
         * @return a copy with GOSS on
         */
        public Settings withGoss(double a, double b) {
            return new Settings(trees, maxLeaves, maxDepth, learningRate, lambda, minChildWeight, minDataInLeaf, bins,
                    a, b, bundle, maxConflictRate, leafWise, seed);
        }

        /**
         * @param on bundling on or off
         * @param conflictRate the tolerated clash share
         * @return a copy with that bundling
         */
        public Settings withBundling(boolean on, double conflictRate) {
            return new Settings(trees, maxLeaves, maxDepth, learningRate, lambda, minChildWeight, minDataInLeaf, bins,
                    topRate, otherRate, on, conflictRate, leafWise, seed);
        }

        /**
         * @param on leaf-wise (true) or level-wise (false)
         * @return a copy with that growth order
         */
        public Settings withLeafWise(boolean on) {
            return new Settings(trees, maxLeaves, maxDepth, learningRate, lambda, minChildWeight, minDataInLeaf, bins,
                    topRate, otherRate, bundle, maxConflictRate, on, seed);
        }

        /**
         * @param rate the shrinkage
         * @return a copy with that learning rate
         */
        public Settings withLearningRate(double rate) {
            return new Settings(trees, maxLeaves, maxDepth, rate, lambda, minChildWeight, minDataInLeaf, bins,
                    topRate, otherRate, bundle, maxConflictRate, leafWise, seed);
        }

        /**
         * @param rows the fewest rows per leaf
         * @return a copy with that minimum
         */
        public Settings withMinDataInLeaf(int rows) {
            return new Settings(trees, maxLeaves, maxDepth, learningRate, lambda, minChildWeight, rows, bins,
                    topRate, otherRate, bundle, maxConflictRate, leafWise, seed);
        }
    }

    /** A tree in flat arrays; node {@code k} splits when {@code feature[k] >= 0}. */
    private record Tree(int[] feature, double[] threshold, int[] left, int[] right, double[] value) {
        double predict(double[] x) {
            int n = 0;
            while (feature[n] >= 0) {
                n = x[feature[n]] < threshold[n] ? left[n] : right[n];
            }
            return value[n];
        }
    }

    private final GradientBoostedTrees.Loss loss;
    private final double base;
    private final List<Tree> trees;
    private final double learningRate;
    private final int bundles;
    private final int leaves;

    private LightBoost(GradientBoostedTrees.Loss loss, double base, List<Tree> trees, double learningRate,
            int bundles, int leaves) {
        this.loss = loss;
        this.base = base;
        this.trees = trees;
        this.learningRate = learningRate;
        this.bundles = bundles;
        this.leaves = leaves;
    }

    /**
     * Trains from the loss's constant starting score.
     *
     * @param x rows of features
     * @param y targets
     * @param loss the loss's derivatives
     * @param s the settings
     * @return the ensemble
     */
    public static LightBoost fit(double[][] x, double[] y, GradientBoostedTrees.Loss loss, Settings s) {
        return fit(x, y, null, loss, s);
    }

    /**
     * Trains on top of per-row starting scores.
     *
     * @param x rows of features
     * @param y targets
     * @param margin per-row starting raw scores, or null for the loss's constant
     * @param loss the loss's derivatives
     * @param s the settings
     * @return the ensemble; pass the same kind of margin to {@link #raw(double[], double)}
     */
    public static LightBoost fit(double[][] x, double[] y, double[] margin, GradientBoostedTrees.Loss loss,
            Settings s) {
        int n = x.length;
        if (n == 0 || y.length != n || (margin != null && margin.length != n)) {
            throw new IllegalArgumentException(n + " rows, " + y.length + " targets");
        }
        int features = x[0].length;
        double base = margin == null ? loss.base(y) : 0;
        double[][] thresholds = new double[features][];
        int[][] bin = new int[features][n];
        for (int f = 0; f < features; f++) {
            thresholds[f] = thresholds(x, f, s.bins());
            for (int i = 0; i < n; i++) {
                bin[f][i] = binOf(thresholds[f], x[i][f]);
            }
        }
        Layout layout = Layout.of(bin, thresholds, s.bundle(), s.maxConflictRate());
        double[] raw = new double[n];
        for (int i = 0; i < n; i++) {
            raw[i] = margin == null ? base : margin[i];
        }
        double[] g = new double[n];
        double[] h = new double[n];
        SplittableRandom random = new SplittableRandom(s.seed());
        List<Tree> trees = new ArrayList<>();
        int leafCount = 0;
        for (int t = 0; t < s.trees(); t++) {
            loss.derivatives(raw, y, g, h);
            double[] gw = g.clone();
            double[] hw = h.clone();
            int[] rows = s.topRate() > 0 ? goss(g, gw, hw, s.topRate(), s.otherRate(), random) : all(n);
            Grower grower = new Grower(layout, thresholds, gw, hw, s);
            Tree tree = grower.grow(rows);
            leafCount += grower.leafCount;
            trees.add(tree);
            for (int i = 0; i < n; i++) {
                raw[i] += s.learningRate() * tree.predict(x[i]);
            }
        }
        return new LightBoost(loss, base, List.copyOf(trees), s.learningRate(), layout.bundles, leafCount);
    }

    private static int[] all(int n) {
        int[] rows = new int[n];
        Arrays.setAll(rows, i -> i);
        return rows;
    }

    /**
     * GOSS: the rows whose |gradient| is in the top {@code a} share, plus each other row with
     * probability {@code b / (1 - a)} (so about {@code b * n} of them), up-weighted by
     * {@code (1 - a) / b}.
     */
    private static int[] goss(double[] g, double[] gw, double[] hw, double a, double b, SplittableRandom random) {
        int n = g.length;
        double[] abs = new double[n];
        for (int i = 0; i < n; i++) {
            abs[i] = Math.abs(g[i]);
        }
        double[] sorted = abs.clone();
        Arrays.sort(sorted);
        int top = (int) Math.ceil(a * n);
        double cut = sorted[Math.clamp(n - top, 0, n - 1)];
        double keep = Math.min(1, b / Math.max(1e-12, 1 - a));
        double amplify = (1 - a) / b;
        int[] rows = new int[n];
        int k = 0;
        int taken = 0;
        for (int i = 0; i < n; i++) {
            if (abs[i] >= cut && taken < top) {
                rows[k++] = i;
                taken++;
            } else if (random.nextDouble() < keep) {
                gw[i] *= amplify;
                hw[i] *= amplify;
                rows[k++] = i;
            }
        }
        return Arrays.copyOf(rows, k);
    }

    private static int binOf(double[] th, double v) {
        int b = Arrays.binarySearch(th, v);
        return b >= 0 ? b + 1 : -b - 1;
    }

    /** Midpoints between quantiles of a feature's distinct values, as the core booster uses. */
    static double[] thresholds(double[][] x, int f, int bins) {
        double[] distinct = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            distinct[i] = x[i][f];
        }
        distinct = Arrays.stream(distinct).sorted().distinct().toArray();
        if (distinct.length <= 1) {
            return new double[0];
        }
        int count = Math.min(bins, distinct.length - 1);
        double[] t = new double[count];
        for (int k = 0; k < count; k++) {
            int at = (int) ((long) (k + 1) * (distinct.length - 1) / (count + 1));
            at = Math.min(at, distinct.length - 2);
            t[k] = (distinct[at] + distinct[at + 1]) / 2;
        }
        return Arrays.stream(t).distinct().toArray();
    }

    /**
     * Where each feature's bins live in the histogram: a bundle (column of the binned matrix), an
     * offset inside the global histogram, and for bundled features the default bin that is not
     * stored.
     */
    private static final class Layout {
        final int[][] columns;      // [bundle][row] -> global histogram slot
        final int[] featureOffset;  // global slot of the feature's bin 0
        final int[] featureBins;    // bins per feature
        final int[] defaultBin;     // -1 when every bin is stored
        final int[] bundleOf;
        final int slots;
        final int bundles;

        private Layout(int[][] columns, int[] featureOffset, int[] featureBins, int[] defaultBin, int[] bundleOf,
                int slots) {
            this.columns = columns;
            this.featureOffset = featureOffset;
            this.featureBins = featureBins;
            this.defaultBin = defaultBin;
            this.bundleOf = bundleOf;
            this.slots = slots;
            this.bundles = columns.length;
        }

        static Layout of(int[][] bin, double[][] thresholds, boolean bundle, double conflictRate) {
            int features = bin.length;
            int n = features == 0 ? 0 : bin[0].length;
            int[] featureBins = new int[features];
            int[] defaultBin = new int[features];
            int[] nonDefault = new int[features];
            for (int f = 0; f < features; f++) {
                featureBins[f] = thresholds[f].length + 1;
                int[] counts = new int[featureBins[f]];
                for (int i = 0; i < n; i++) {
                    counts[bin[f][i]]++;
                }
                int best = 0;
                for (int b = 1; b < counts.length; b++) {
                    if (counts[b] > counts[best]) {
                        best = b;
                    }
                }
                defaultBin[f] = best;
                nonDefault[f] = n - counts[best];
            }
            List<List<Integer>> groups = new ArrayList<>();
            if (bundle) {
                Integer[] order = new Integer[features];
                Arrays.setAll(order, i -> i);
                Arrays.sort(order, (p, q) -> Integer.compare(nonDefault[q], nonDefault[p]));
                List<boolean[]> used = new ArrayList<>();
                List<int[]> conflicts = new ArrayList<>();
                int allowed = (int) (conflictRate * n);
                for (int f : order) {
                    int placed = -1;
                    for (int g = 0; g < groups.size() && placed < 0; g++) {
                        boolean[] u = used.get(g);
                        int c = 0;
                        for (int i = 0; i < n && c <= allowed; i++) {
                            if (u[i] && bin[f][i] != defaultBin[f]) {
                                c++;
                            }
                        }
                        if (conflicts.get(g)[0] + c <= allowed) {
                            placed = g;
                            conflicts.get(g)[0] += c;
                        }
                    }
                    if (placed < 0) {
                        groups.add(new ArrayList<>());
                        used.add(new boolean[n]);
                        conflicts.add(new int[1]);
                        placed = groups.size() - 1;
                    }
                    groups.get(placed).add(f);
                    boolean[] u = used.get(placed);
                    for (int i = 0; i < n; i++) {
                        if (bin[f][i] != defaultBin[f]) {
                            u[i] = true;
                        }
                    }
                }
            } else {
                for (int f = 0; f < features; f++) {
                    groups.add(List.of(f));
                }
            }
            int[][] columns = new int[groups.size()][n];
            int[] featureOffset = new int[features];
            int[] bundleOf = new int[features];
            int[] storedDefault = new int[features];
            int slot = 0;
            for (int g = 0; g < groups.size(); g++) {
                List<Integer> members = groups.get(g);
                if (members.size() == 1) {
                    int f = members.get(0);
                    featureOffset[f] = slot;
                    bundleOf[f] = g;
                    storedDefault[f] = -1;
                    for (int i = 0; i < n; i++) {
                        columns[g][i] = slot + bin[f][i];
                    }
                    slot += featureBins[f];
                    continue;
                }
                int zero = slot++;   // every member at its default bin
                Arrays.fill(columns[g], zero);
                for (int f : members) {
                    featureOffset[f] = slot;
                    bundleOf[f] = g;
                    storedDefault[f] = defaultBin[f];
                    for (int i = 0; i < n; i++) {
                        if (bin[f][i] != defaultBin[f] && columns[g][i] == zero) {
                            columns[g][i] = slot + bin[f][i];
                        }
                    }
                    slot += featureBins[f];
                }
            }
            return new Layout(columns, featureOffset, featureBins, storedDefault, bundleOf, slot);
        }
    }

    /** One tree's growth: leaves with histograms, split one at a time. */
    private static final class Grower {
        final Layout layout;
        final double[][] thresholds;
        final double[] g;
        final double[] h;
        final Settings s;
        final List<int[]> nodes = new ArrayList<>();      // feature, left, right
        final List<double[]> values = new ArrayList<>();  // threshold, value
        int leafCount;

        Grower(Layout layout, double[][] thresholds, double[] g, double[] h, Settings s) {
            this.layout = layout;
            this.thresholds = thresholds;
            this.g = g;
            this.h = h;
            this.s = s;
        }

        /** A leaf being grown. */
        final class Leaf {
            final int node;
            final int[] rows;
            final double[] hist;   // [3 * slots]: gradient, curvature, count
            final int depth;
            final double gs;
            final double hs;
            int feature = -1;
            int cut;
            double gain;

            Leaf(int node, int[] rows, double[] hist, int depth) {
                this.node = node;
                this.rows = rows;
                this.hist = hist;
                this.depth = depth;
                double a = 0;
                double b = 0;
                for (int r : rows) {
                    a += g[r];
                    b += h[r];
                }
                gs = a;
                hs = b;
            }
        }

        double[] histogram(int[] rows) {
            int slots = layout.slots;
            double[] hist = new double[3 * slots];
            for (int[] column : layout.columns) {
                for (int r : rows) {
                    int k = column[r];
                    hist[k] += g[r];
                    hist[slots + k] += h[r];
                    hist[2 * slots + k] += 1;
                }
            }
            return hist;
        }

        int node() {
            nodes.add(new int[] {-1, -1, -1});
            values.add(new double[2]);
            return nodes.size() - 1;
        }

        void findSplit(Leaf leaf) {
            if (s.maxDepth() > 0 && leaf.depth >= s.maxDepth() || leaf.rows.length < 2 * s.minDataInLeaf()) {
                return;
            }
            int slots = layout.slots;
            double parent = leaf.gs * leaf.gs / (leaf.hs + s.lambda());
            double best = 1e-12;
            double[] fg = new double[0];
            double[] fh = new double[0];
            double[] fc = new double[0];
            for (int f = 0; f < thresholds.length; f++) {
                int bins = layout.featureBins[f];
                if (bins < 2) {
                    continue;
                }
                if (fg.length < bins) {
                    fg = new double[bins];
                    fh = new double[bins];
                    fc = new double[bins];
                }
                int off = layout.featureOffset[f];
                int def = layout.defaultBin[f];
                double rg = leaf.gs;
                double rh = leaf.hs;
                double rc = leaf.rows.length;
                for (int b = 0; b < bins; b++) {
                    if (b == def) {
                        continue;
                    }
                    fg[b] = leaf.hist[off + b];
                    fh[b] = leaf.hist[slots + off + b];
                    fc[b] = leaf.hist[2 * slots + off + b];
                    rg -= fg[b];
                    rh -= fh[b];
                    rc -= fc[b];
                }
                if (def >= 0) {
                    fg[def] = rg;
                    fh[def] = rh;
                    fc[def] = rc;
                }
                double gl = 0;
                double hl = 0;
                double cl = 0;
                for (int k = 0; k + 1 < bins; k++) {
                    gl += fg[k];
                    hl += fh[k];
                    cl += fc[k];
                    double gr = leaf.gs - gl;
                    double hr = leaf.hs - hl;
                    double cr = leaf.rows.length - cl;
                    if (cl < s.minDataInLeaf() || cr < s.minDataInLeaf() || hl < s.minChildWeight()
                            || hr < s.minChildWeight()) {
                        continue;
                    }
                    double gain = gl * gl / (hl + s.lambda()) + gr * gr / (hr + s.lambda()) - parent;
                    if (gain > best) {
                        best = gain;
                        leaf.feature = f;
                        leaf.cut = k;
                    }
                }
            }
            leaf.gain = best;
        }

        Tree grow(int[] rows) {
            List<Leaf> open = new ArrayList<>();
            Leaf root = new Leaf(node(), rows, histogram(rows), 0);
            findSplit(root);
            open.add(root);
            List<Leaf> done = new ArrayList<>();
            int count = 1;
            while (count < s.maxLeaves()) {
                Leaf pick = null;
                for (Leaf l : open) {
                    if (l.feature < 0) {
                        continue;
                    }
                    if (pick == null || (s.leafWise() ? l.gain > pick.gain : l.depth < pick.depth)) {
                        pick = l;
                    }
                }
                if (pick == null) {
                    break;
                }
                open.remove(pick);
                int f = pick.feature;
                double t = thresholds[f][pick.cut];
                int cut = pick.cut;
                int[] binOf = layout.columns[layout.bundleOf[f]];
                int off = layout.featureOffset[f];
                int def = layout.defaultBin[f];
                int leftCount = 0;
                boolean[] goLeft = new boolean[pick.rows.length];
                for (int i = 0; i < pick.rows.length; i++) {
                    int slot = binOf[pick.rows[i]] - off;
                    int b = slot >= 0 && slot < layout.featureBins[f] && slot != def ? slot : (def >= 0 ? def : slot);
                    goLeft[i] = b <= cut;
                    leftCount += goLeft[i] ? 1 : 0;
                }
                int[] left = new int[leftCount];
                int[] right = new int[pick.rows.length - leftCount];
                for (int i = 0, a = 0, c = 0; i < pick.rows.length; i++) {
                    if (goLeft[i]) {
                        left[a++] = pick.rows[i];
                    } else {
                        right[c++] = pick.rows[i];
                    }
                }
                boolean leftSmaller = left.length <= right.length;
                double[] small = histogram(leftSmaller ? left : right);
                double[] large = pick.hist.clone();
                for (int k = 0; k < large.length; k++) {
                    large[k] -= small[k];
                }
                Leaf l = new Leaf(node(), left, leftSmaller ? small : large, pick.depth + 1);
                Leaf r = new Leaf(node(), right, leftSmaller ? large : small, pick.depth + 1);
                nodes.set(pick.node, new int[] {f, l.node, r.node});
                values.get(pick.node)[0] = t;
                findSplit(l);
                findSplit(r);
                open.add(l);
                open.add(r);
                count++;
            }
            done.addAll(open);
            for (Leaf l : done) {
                values.get(l.node)[1] = -l.gs / (l.hs + s.lambda());
            }
            leafCount = count;
            int n = nodes.size();
            int[] feature = new int[n];
            int[] lft = new int[n];
            int[] rgt = new int[n];
            double[] threshold = new double[n];
            double[] value = new double[n];
            for (int i = 0; i < n; i++) {
                feature[i] = nodes.get(i)[0];
                lft[i] = nodes.get(i)[1];
                rgt[i] = nodes.get(i)[2];
                threshold[i] = values.get(i)[0];
                value[i] = values.get(i)[1];
            }
            return new Tree(feature, threshold, lft, rgt, value);
        }
    }

    /**
     * @param x a row
     * @return the raw score from the constant start
     */
    public double raw(double[] x) {
        return raw(x, base);
    }

    /**
     * @param x a row
     * @param margin the row's starting score (as passed to {@code fit} for training rows)
     * @return the raw score
     */
    public double raw(double[] x, double margin) {
        double r = margin;
        for (Tree t : trees) {
            r += learningRate * t.predict(x);
        }
        return r;
    }

    /**
     * @param x a row
     * @return the loss's link of the raw score
     */
    public double predict(double[] x) {
        return loss.link(raw(x));
    }

    /**
     * @param x rows
     * @return one prediction per row
     */
    public double[] predict(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = predict(x[i]);
        }
        return out;
    }

    /** @return the number of histogram columns after bundling */
    public int bundles() {
        return bundles;
    }

    /** @return the total number of leaves over all trees */
    public int leaves() {
        return leaves;
    }

    /** @return the number of trees */
    public int size() {
        return trees.size();
    }
}
