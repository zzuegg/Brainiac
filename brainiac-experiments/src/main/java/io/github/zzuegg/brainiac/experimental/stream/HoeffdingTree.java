package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.experimental.data.DriftingStreams;

import java.util.SplittableRandom;

/**
 * A Hoeffding tree (VFDT): a decision tree grown from a stream in one pass with constant memory
 * per leaf.
 *
 * <p>Each leaf keeps sufficient statistics: class weights and, per numeric attribute and class,
 * a Gaussian (count, mean, variance, range). Every {@code grace} examples a leaf evaluates
 * candidate splits (ten thresholds per attribute, class masses on each side estimated from the
 * Gaussians) by information gain. By the Hoeffding bound
 * {@code eps = sqrt(R^2 ln(1/delta) / (2 n))}, {@code R = log2(classes)}, the best attribute is
 * truly better than the second with probability {@code 1 - delta} once their gains differ by
 * more than {@code eps}, and the leaf splits; ties are broken once {@code eps < tau}. Leaves
 * predict with the better of majority class and Gaussian naive Bayes, chosen by their own
 * running accuracy (MOA's "NB adaptive" leaves).
 *
 * <p>With {@code subspace > 0} each new leaf considers a random subset of that many attributes,
 * as the trees of an {@link AdaptiveRandomForest} do. {@link #learn(double[], int, double)} takes
 * example weights for online bagging.
 *
 * <p>Paper: Domingos and Hulten, "Mining High-Speed Data Streams", KDD 2000,
 * <a href="https://doi.org/10.1145/347090.347107">doi:10.1145/347090.347107</a>.
 */
public final class HoeffdingTree implements OnlineClassifier {

    private final int features;
    private final int classes;
    private final double delta;
    private final double tau;
    private final int grace;
    private final int subspace;
    private final int maxDepth;
    private final SplittableRandom random;
    private Node root;
    private int leaves = 1;
    private int splits;

    private abstract static class Node {
        final int depth;

        Node(int depth) {
            this.depth = depth;
        }
    }

    private static final class Split extends Node {
        final int attribute;
        final double threshold;
        Node left;
        Node right;

        Split(int depth, int attribute, double threshold, Node left, Node right) {
            super(depth);
            this.attribute = attribute;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
        }
    }

    private final class Leaf extends Node {
        final double[] classWeight = new double[classes];
        final double[][] n;
        final double[][] mean;
        final double[][] m2;
        final double[][] min;
        final double[][] max;
        final int[] attributes;
        double seenAtLastCheck;
        double total;
        double majorityCorrect;
        double bayesCorrect;

        Leaf(int depth) {
            super(depth);
            n = new double[features][classes];
            mean = new double[features][classes];
            m2 = new double[features][classes];
            min = new double[features][classes];
            max = new double[features][classes];
            for (int a = 0; a < features; a++) {
                java.util.Arrays.fill(min[a], Double.POSITIVE_INFINITY);
                java.util.Arrays.fill(max[a], Double.NEGATIVE_INFINITY);
            }
            if (subspace > 0 && subspace < features) {
                int[] all = new int[features];
                for (int i = 0; i < features; i++) {
                    all[i] = i;
                }
                for (int i = 0; i < subspace; i++) {
                    int j = i + random.nextInt(features - i);
                    int t = all[i];
                    all[i] = all[j];
                    all[j] = t;
                }
                attributes = java.util.Arrays.copyOf(all, subspace);
            } else {
                attributes = null;
            }
        }

        double[] majority() {
            double[] p = new double[classes];
            double s = 0;
            for (double w : classWeight) {
                s += w;
            }
            for (int c = 0; c < classes; c++) {
                p[c] = s == 0 ? 1.0 / classes : (classWeight[c] + 1e-3) / (s + 1e-3 * classes);
            }
            return p;
        }

        double[] bayes(double[] x) {
            double[] logp = new double[classes];
            double s = 0;
            for (double w : classWeight) {
                s += w;
            }
            double best = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                double lp = Math.log((classWeight[c] + 1) / (s + classes));
                for (int a = 0; a < features; a++) {
                    if (n[a][c] < 2) {
                        continue;
                    }
                    double var = Math.max(m2[a][c] / n[a][c], 1e-6);
                    double d = x[a] - mean[a][c];
                    lp += -0.5 * Math.log(2 * Math.PI * var) - d * d / (2 * var);
                }
                logp[c] = lp;
                best = Math.max(best, lp);
            }
            double z = 0;
            for (int c = 0; c < classes; c++) {
                logp[c] = Math.exp(logp[c] - best);
                z += logp[c];
            }
            for (int c = 0; c < classes; c++) {
                logp[c] /= z;
            }
            return logp;
        }

        double[] predict(double[] x) {
            return bayesCorrect > majorityCorrect ? bayes(x) : majority();
        }

        void learn(double[] x, int y, double w) {
            if (total > 0) {
                majorityCorrect += OnlineClassifier.argmax(majority()) == y ? w : 0;
                bayesCorrect += OnlineClassifier.argmax(bayes(x)) == y ? w : 0;
            }
            classWeight[y] += w;
            total += w;
            for (int a = 0; a < features; a++) {
                double nn = n[a][y] + w;
                double d = x[a] - mean[a][y];
                mean[a][y] += d * w / nn;
                m2[a][y] += w * d * (x[a] - mean[a][y]);
                n[a][y] = nn;
                min[a][y] = Math.min(min[a][y], x[a]);
                max[a][y] = Math.max(max[a][y], x[a]);
            }
        }
    }

    /**
     * Creates a tree.
     *
     * @param features numeric attributes
     * @param classes classes
     * @param delta the split confidence, such as 1e-7
     * @param tau the tie threshold, such as 0.05
     * @param grace examples between split attempts at a leaf, such as 200
     * @param subspace attributes per leaf (0 for all)
     * @param maxDepth the depth limit
     * @param seed the seed for the random subspaces
     */
    public HoeffdingTree(int features, int classes, double delta, double tau, int grace, int subspace, int maxDepth,
            long seed) {
        this.features = features;
        this.classes = classes;
        this.delta = delta;
        this.tau = tau;
        this.grace = grace;
        this.subspace = subspace;
        this.maxDepth = maxDepth;
        this.random = new SplittableRandom(seed);
        this.root = new Leaf(0);
    }

    /**
     * A tree with the usual defaults: {@code delta = 1e-7}, {@code tau = 0.05}, grace 200, all
     * attributes, depth 20.
     *
     * @param features numeric attributes
     * @param classes classes
     * @return the tree
     */
    public static HoeffdingTree of(int features, int classes) {
        return new HoeffdingTree(features, classes, 1e-7, 0.05, 200, 0, 20, 1);
    }

    private Leaf leaf(double[] x) {
        Node node = root;
        while (node instanceof Split s) {
            node = x[s.attribute] <= s.threshold ? s.left : s.right;
        }
        return (Leaf) node;
    }

    /**
     * Class probabilities.
     *
     * @param x the attributes
     * @return probabilities
     */
    public double[] predict(double[] x) {
        return leaf(x).predict(x);
    }

    @Override
    public double[] predict(DriftingStreams.Sample sample) {
        return predict(sample.x());
    }

    @Override
    public void learn(DriftingStreams.Sample sample) {
        learn(sample.x(), sample.label(), 1);
    }

    /**
     * Learns from a weighted example.
     *
     * @param x the attributes
     * @param y the class
     * @param weight the example weight (0 skips it)
     */
    public void learn(double[] x, int y, double weight) {
        if (weight <= 0) {
            return;
        }
        Split parent = null;
        boolean left = false;
        Node node = root;
        while (node instanceof Split s) {
            parent = s;
            left = x[s.attribute] <= s.threshold;
            node = left ? s.left : s.right;
        }
        Leaf l = (Leaf) node;
        l.learn(x, y, weight);
        if (l.total - l.seenAtLastCheck >= grace && l.depth < maxDepth) {
            l.seenAtLastCheck = l.total;
            Split s = attemptSplit(l);
            if (s != null) {
                if (parent == null) {
                    root = s;
                } else if (left) {
                    parent.left = s;
                } else {
                    parent.right = s;
                }
                leaves++;
                splits++;
            }
        }
    }

    private static double entropy(double[] w) {
        double s = 0;
        for (double v : w) {
            s += v;
        }
        if (s <= 0) {
            return 0;
        }
        double h = 0;
        for (double v : w) {
            if (v > 0) {
                h -= v / s * Math.log(v / s) / Math.log(2);
            }
        }
        return h;
    }

    private static double normalCdf(double z) {
        double t = 1 / (1 + 0.2316419 * Math.abs(z));
        double d = 0.3989423 * Math.exp(-z * z / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return z > 0 ? 1 - p : p;
    }

    private Split attemptSplit(Leaf l) {
        int nonEmpty = 0;
        for (double w : l.classWeight) {
            nonEmpty += w > 0 ? 1 : 0;
        }
        if (nonEmpty < 2) {
            return null;
        }
        double parentH = entropy(l.classWeight);
        double best = 0;
        double second = 0;
        int bestA = -1;
        double bestT = 0;
        int[] attrs = l.attributes;
        int count = attrs == null ? features : attrs.length;
        for (int k = 0; k < count; k++) {
            int a = attrs == null ? k : attrs[k];
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                lo = Math.min(lo, l.min[a][c]);
                hi = Math.max(hi, l.max[a][c]);
            }
            if (!(hi > lo)) {
                continue;
            }
            double attrBest = 0;
            double attrT = 0;
            for (int s = 1; s <= 10; s++) {
                double t = lo + (hi - lo) * s / 11.0;
                double[] lw = new double[classes];
                double[] rw = new double[classes];
                for (int c = 0; c < classes; c++) {
                    double nc = l.n[a][c];
                    if (nc == 0) {
                        continue;
                    }
                    double sd = Math.sqrt(Math.max(l.m2[a][c] / nc, 1e-12));
                    double frac = t < l.min[a][c] ? 0 : t >= l.max[a][c] ? 1 : normalCdf((t - l.mean[a][c]) / sd);
                    lw[c] = nc * frac;
                    rw[c] = nc - lw[c];
                }
                double ln = 0;
                double rn = 0;
                for (int c = 0; c < classes; c++) {
                    ln += lw[c];
                    rn += rw[c];
                }
                double tot = ln + rn;
                if (ln < 0.01 * tot || rn < 0.01 * tot) {
                    continue;
                }
                double gain = parentH - (ln / tot) * entropy(lw) - (rn / tot) * entropy(rw);
                if (gain > attrBest) {
                    attrBest = gain;
                    attrT = t;
                }
            }
            if (attrBest > best) {
                second = best;
                best = attrBest;
                bestA = a;
                bestT = attrT;
            } else if (attrBest > second) {
                second = attrBest;
            }
        }
        if (bestA < 0) {
            return null;
        }
        double range = Math.log(classes) / Math.log(2);
        double eps = Math.sqrt(range * range * Math.log(1 / delta) / (2 * l.total));
        if (best - second > eps || eps < tau) {
            return new Split(l.depth, bestA, bestT, new Leaf(l.depth + 1), new Leaf(l.depth + 1));
        }
        return null;
    }

    /** @return the number of leaves */
    public int leaves() {
        return leaves;
    }

    /** @return the number of splits made */
    public int splits() {
        return splits;
    }

    /** Leaves hold five doubles per attribute and class plus the class weights; splits 16 bytes. */
    @Override
    public long bytes() {
        return (long) leaves * (5L * features * classes + classes) * 8 + 16L * splits;
    }
}
