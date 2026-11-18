package io.github.zzuegg.brainiac.experimental.tsc;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Proximity Forest (Lucas et al., DMKD 2019, <a href="https://arxiv.org/abs/1808.10594">arXiv:1808.10594</a>)
 * and the Elastic Ensemble it was built to replace (Lines and Bagnall, DMKD 2015).
 *
 * <p>A proximity tree splits a node by picking one random exemplar per class from the node's
 * series, one random elastic measure with random parameters ({@link ElasticDistances.Measure}),
 * and sending every series to the branch of its nearest exemplar. Of {@code candidates} such random
 * splits the one with the largest Gini gain wins; nodes stop when pure. The forest votes. No
 * cross-validation is needed, which is where the speed-up over the Elastic Ensemble comes from.
 *
 * <p>The {@link #elasticEnsemble Elastic Ensemble} instead tunes each measure's parameters by
 * leave-one-out 1-NN accuracy over a grid, and weights each measure's 1-NN vote by that accuracy.
 */
public final class ProximityForest {

    /** A node: a leaf with a class or a split with exemplars, measure and children. */
    private static final class Node {
        int label = -1;
        double[][] exemplars;
        int[] branchLabel;
        ElasticDistances.Measure measure;
        double[] params;
        Node[] children;
    }

    private final List<Node> trees;
    private final int classes;

    private ProximityForest(List<Node> trees, int classes) {
        this.trees = trees;
        this.classes = classes;
    }

    /**
     * Grows a forest.
     *
     * @param series training series
     * @param labels their classes
     * @param classes the number of classes
     * @param trees the number of trees, such as 50 to 100
     * @param candidates random splits tried per node, 5 in the paper
     * @param measures the measures to draw from
     * @param seed the seed
     * @return the forest
     */
    public static ProximityForest fit(double[][] series, int[] labels, int classes, int trees, int candidates,
            ElasticDistances.Measure[] measures, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double sd = sd(series);
        List<Node> out = new ArrayList<>();
        int[] all = new int[series.length];
        for (int i = 0; i < all.length; i++) {
            all[i] = i;
        }
        for (int t = 0; t < trees; t++) {
            out.add(grow(series, labels, classes, all, candidates, measures, sd, r, 0));
        }
        return new ProximityForest(out, classes);
    }

    /**
     * The standard deviation of all training values.
     *
     * @param series the series
     * @return the pooled standard deviation
     */
    public static double sd(double[][] series) {
        double s = 0;
        double s2 = 0;
        long n = 0;
        for (double[] x : series) {
            for (double v : x) {
                s += v;
                s2 += v * v;
                n++;
            }
        }
        double mean = s / n;
        return Math.sqrt(Math.max(1e-12, s2 / n - mean * mean));
    }

    private static Node grow(double[][] x, int[] y, int classes, int[] rows, int candidates,
            ElasticDistances.Measure[] measures, double sd, SplittableRandom r, int depth) {
        Node node = new Node();
        int[] counts = new int[classes];
        for (int i : rows) {
            counts[y[i]]++;
        }
        int major = 0;
        int present = 0;
        for (int c = 0; c < classes; c++) {
            if (counts[c] > counts[major]) {
                major = c;
            }
            present += counts[c] > 0 ? 1 : 0;
        }
        if (present <= 1 || depth > 30) {
            node.label = major;
            return node;
        }
        double parentGini = gini(counts, rows.length);
        double bestGain = Double.NEGATIVE_INFINITY;
        int[][] bestBranches = null;
        for (int cand = 0; cand < candidates; cand++) {
            List<double[]> ex = new ArrayList<>();
            List<Integer> exLabel = new ArrayList<>();
            for (int c = 0; c < classes; c++) {
                if (counts[c] == 0) {
                    continue;
                }
                int pick = r.nextInt(counts[c]);
                for (int i : rows) {
                    if (y[i] == c && pick-- == 0) {
                        ex.add(x[i]);
                        exLabel.add(c);
                        break;
                    }
                }
            }
            ElasticDistances.Measure m = measures[r.nextInt(measures.length)];
            double[] p = m.random(sd, r);
            int[] branch = new int[rows.length];
            int[][] branchCounts = new int[ex.size()][classes];
            int[] branchSize = new int[ex.size()];
            for (int k = 0; k < rows.length; k++) {
                int b = nearest(x[rows[k]], ex, m, p, r);
                branch[k] = b;
                branchCounts[b][y[rows[k]]]++;
                branchSize[b]++;
            }
            double child = 0;
            for (int b = 0; b < ex.size(); b++) {
                child += branchSize[b] / (double) rows.length * gini(branchCounts[b], branchSize[b]);
            }
            double gain = parentGini - child;
            if (gain > bestGain) {
                bestGain = gain;
                node.exemplars = ex.toArray(new double[0][]);
                node.branchLabel = exLabel.stream().mapToInt(Integer::intValue).toArray();
                node.measure = m;
                node.params = p;
                int[][] br = new int[ex.size()][];
                for (int b = 0; b < ex.size(); b++) {
                    br[b] = new int[branchSize[b]];
                }
                int[] fill = new int[ex.size()];
                for (int k = 0; k < rows.length; k++) {
                    br[branch[k]][fill[branch[k]]++] = rows[k];
                }
                bestBranches = br;
            }
        }
        node.children = new Node[node.exemplars.length];
        for (int b = 0; b < node.children.length; b++) {
            if (bestBranches[b].length == 0) {
                Node leaf = new Node();
                leaf.label = node.branchLabel[b];
                node.children[b] = leaf;
            } else if (bestBranches[b].length == rows.length) {
                Node leaf = new Node();
                leaf.label = major;
                node.children[b] = leaf;
            } else {
                node.children[b] = grow(x, y, classes, bestBranches[b], candidates, measures, sd, r, depth + 1);
            }
        }
        return node;
    }

    private static int nearest(double[] q, List<double[]> ex, ElasticDistances.Measure m, double[] p,
            SplittableRandom r) {
        double best = Double.POSITIVE_INFINITY;
        int arg = 0;
        int ties = 0;
        for (int e = 0; e < ex.size(); e++) {
            double d = m.distance(q, ex.get(e), p);
            if (d < best) {
                best = d;
                arg = e;
                ties = 1;
            } else if (d == best && r.nextInt(++ties) == 0) {
                arg = e;
            }
        }
        return arg;
    }

    private static double gini(int[] counts, int n) {
        if (n == 0) {
            return 0;
        }
        double s = 1;
        for (int c : counts) {
            double p = c / (double) n;
            s -= p * p;
        }
        return s;
    }

    /**
     * Vote shares per class.
     *
     * @param series the series
     * @return {@code [n][classes]}
     */
    public double[][] probabilities(double[][] series) {
        double[][] out = new double[series.length][classes];
        for (int i = 0; i < series.length; i++) {
            for (Node t : trees) {
                Node n = t;
                while (n.label < 0) {
                    double best = Double.POSITIVE_INFINITY;
                    int arg = 0;
                    for (int e = 0; e < n.exemplars.length; e++) {
                        double d = n.measure.distance(series[i], n.exemplars[e], n.params);
                        if (d < best) {
                            best = d;
                            arg = e;
                        }
                    }
                    n = n.children[arg];
                }
                out[i][n.label] += 1.0 / trees.size();
            }
        }
        return out;
    }

    /**
     * Predicts classes.
     *
     * @param series the series
     * @return the classes
     */
    public int[] predict(double[][] series) {
        return WideRidgeClassifier.argmax(probabilities(series));
    }

    /**
     * 1-nearest-neighbour classification under one measure.
     *
     * @param train training series
     * @param labels their classes
     * @param test series to classify
     * @param m the measure
     * @param p its parameters
     * @return the predicted classes
     */
    public static int[] nearestNeighbour(double[][] train, int[] labels, double[][] test, ElasticDistances.Measure m,
            double[] p) {
        int[] out = new int[test.length];
        for (int i = 0; i < test.length; i++) {
            double best = Double.POSITIVE_INFINITY;
            for (int j = 0; j < train.length; j++) {
                double d = m.distance(test[i], train[j], p);
                if (d < best) {
                    best = d;
                    out[i] = labels[j];
                }
            }
        }
        return out;
    }

    /**
     * The result of a pruned 1-NN search.
     *
     * @param predictions the predicted classes
     * @param prunedShare the share of (query, candidate) pairs skipped by the lower bound
     */
    public record PrunedSearch(int[] predictions, double prunedShare) {
    }

    /**
     * 1-NN with windowed DTW, pruned by LB_Keogh before computing the full distance.
     *
     * @param train training series
     * @param labels their classes
     * @param test series to classify
     * @param window the DTW window half-width
     * @return the predictions and the pruned share
     */
    public static PrunedSearch dtwWithLowerBound(double[][] train, int[] labels, double[][] test, int window) {
        int[] out = new int[test.length];
        long pruned = 0;
        for (int i = 0; i < test.length; i++) {
            double best = Double.POSITIVE_INFINITY;
            for (int j = 0; j < train.length; j++) {
                if (ElasticDistances.lbKeogh(test[i], train[j], window) >= best) {
                    pruned++;
                    continue;
                }
                double d = ElasticDistances.dtw(test[i], train[j], window);
                if (d < best) {
                    best = d;
                    out[i] = labels[j];
                }
            }
        }
        return new PrunedSearch(out, pruned / ((double) test.length * train.length));
    }

    /**
     * The Elastic Ensemble: every measure's parameters tuned by leave-one-out 1-NN accuracy over
     * its grid, then a vote weighted by those accuracies.
     *
     * @param train training series
     * @param labels their classes
     * @param classes the number of classes
     * @param test series to classify
     * @param measures the member measures
     * @return vote weights per class {@code [test][classes]}
     */
    public static double[][] elasticEnsemble(double[][] train, int[] labels, int classes, double[][] test,
            ElasticDistances.Measure[] measures) {
        double sd = sd(train);
        double[][] out = new double[test.length][classes];
        for (ElasticDistances.Measure m : measures) {
            double bestAcc = -1;
            double[] bestP = null;
            for (double[] p : m.grid(sd)) {
                int right = 0;
                for (int i = 0; i < train.length; i++) {
                    double best = Double.POSITIVE_INFINITY;
                    int lab = -1;
                    for (int j = 0; j < train.length; j++) {
                        if (j != i) {
                            double d = m.distance(train[i], train[j], p);
                            if (d < best) {
                                best = d;
                                lab = labels[j];
                            }
                        }
                    }
                    right += lab == labels[i] ? 1 : 0;
                }
                double acc = right / (double) train.length;
                if (acc > bestAcc) {
                    bestAcc = acc;
                    bestP = p;
                }
            }
            int[] pred = nearestNeighbour(train, labels, test, m, bestP);
            for (int i = 0; i < test.length; i++) {
                out[i][pred[i]] += bestAcc;
            }
        }
        return out;
    }
}
