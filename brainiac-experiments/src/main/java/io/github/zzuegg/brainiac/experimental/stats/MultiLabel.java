package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Multi-label classification with logistic base learners: every input has a subset of {@code L}
 * labels.
 *
 * <ul>
 *   <li>{@link Method#BINARY_RELEVANCE}: one independent classifier per label (ignores label
 *       correlations).</li>
 *   <li>{@link Method#CLASSIFIER_CHAIN} (Read, Pfahringer, Holmes and Frank, Machine Learning
 *       2011): classifier {@code j} also sees labels {@code 0 .. j-1} of the chain order (true
 *       labels in training, predictions at test time); with {@link #beamWidth} &gt; 1 the chain
 *       is decoded as a probabilistic chain by beam search over the joint probability, which
 *       targets subset accuracy.</li>
 *   <li>{@link Method#ENSEMBLE_CHAINS}: chains in random orders, marginals averaged.</li>
 *   <li>{@link Method#LABEL_POWERSET} (Tsoumakas and Katakis 2007): every observed label set is
 *       one class (one-vs-rest logistic, argmax).</li>
 *   <li>{@link Method#ML_KNN} (Zhang and Zhou, Pattern Recognition 2007): a Bayesian rule on how
 *       many of the {@code k} nearest neighbours carry each label, with Laplace smoothing.</li>
 * </ul>
 */
public final class MultiLabel {

    /** The method. */
    public enum Method {
        /** Independent per-label classifiers. */
        BINARY_RELEVANCE,
        /** One classifier chain. */
        CLASSIFIER_CHAIN,
        /** Ensemble of chains in random orders. */
        ENSEMBLE_CHAINS,
        /** Label powerset. */
        LABEL_POWERSET,
        /** Multi-label k nearest neighbours. */
        ML_KNN
    }

    private final Method method;
    private final int labels;
    private final List<int[]> orders = new ArrayList<>();
    private final List<double[][]> chains = new ArrayList<>();
    private int beamWidth = 1;
    private List<int[]> powersets;
    private double[][] powersetModels;
    private double[][] trainX;
    private int[][] trainY;
    private int k = 10;
    private double[] prior;
    private double[][][] countGiven;

    private MultiLabel(Method method, int labels) {
        this.method = method;
        this.labels = labels;
    }

    /**
     * Trains a multi-label model.
     *
     * @param method the method
     * @param x features {@code [n][p]}
     * @param y labels {@code [n][L]} in {0, 1}
     * @param members chains for {@link Method#ENSEMBLE_CHAINS}, neighbours for {@link Method#ML_KNN}
     * @param seed the seed for chain orders
     * @return the model
     */
    public static MultiLabel fit(Method method, double[][] x, int[][] y, int members, long seed) {
        int l = y[0].length;
        MultiLabel m = new MultiLabel(method, l);
        SplittableRandom r = new SplittableRandom(seed);
        switch (method) {
            case BINARY_RELEVANCE -> {
                int[] order = new int[l];
                for (int j = 0; j < l; j++) {
                    order[j] = j;
                }
                m.orders.add(order);
                m.chains.add(trainChain(x, y, order, false));
            }
            case CLASSIFIER_CHAIN -> {
                int[] order = new int[l];
                for (int j = 0; j < l; j++) {
                    order[j] = j;
                }
                m.orders.add(order);
                m.chains.add(trainChain(x, y, order, true));
            }
            case ENSEMBLE_CHAINS -> {
                for (int c = 0; c < members; c++) {
                    int[] order = permutation(l, r);
                    m.orders.add(order);
                    m.chains.add(trainChain(x, y, order, true));
                }
            }
            case LABEL_POWERSET -> m.trainPowerset(x, y);
            case ML_KNN -> m.trainKnn(x, y, members);
            default -> throw new IllegalArgumentException(String.valueOf(method));
        }
        return m;
    }

    /**
     * Sets the beam width for chain decoding (1 is greedy).
     *
     * @param width the beam width
     * @return this
     */
    public MultiLabel beamWidth(int width) {
        this.beamWidth = width;
        return this;
    }

    private static double[][] trainChain(double[][] x, int[][] y, int[] order, boolean chained) {
        int n = x.length;
        int p = x[0].length;
        int l = order.length;
        double[][] models = new double[l][];
        for (int a = 0; a < l; a++) {
            int extra = chained ? a : 0;
            double[][] f = new double[n][p + extra + 1];
            double[] t = new double[n];
            for (int i = 0; i < n; i++) {
                System.arraycopy(x[i], 0, f[i], 0, p);
                for (int b = 0; b < extra; b++) {
                    f[i][p + b] = y[i][order[b]];
                }
                f[i][p + extra] = 1;
                t[i] = y[i][order[a]];
            }
            models[a] = Recalibrators.logistic(f, t, 1e-3);
        }
        return models;
    }

    private static double prob(double[] w, double[] row, double[] prev, int extra) {
        int p = row.length;
        double s = w[p + extra];
        for (int j = 0; j < p; j++) {
            s += w[j] * row[j];
        }
        for (int b = 0; b < extra; b++) {
            s += w[p + b] * prev[b];
        }
        return 1 / (1 + Math.exp(-s));
    }

    /**
     * Predicts label sets.
     *
     * @param x rows
     * @return {@code [n][L]} in {0, 1}
     */
    public int[][] predict(double[][] x) {
        int[][] out = new int[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = predict(x[i]);
        }
        return out;
    }

    /**
     * Predicts one label set.
     *
     * @param row the input
     * @return the labels
     */
    public int[] predict(double[] row) {
        return switch (method) {
            case BINARY_RELEVANCE, CLASSIFIER_CHAIN -> beamWidth > 1 && method == Method.CLASSIFIER_CHAIN
                    ? beam(row, orders.get(0), chains.get(0)) : threshold(chainMarginals(row, 0));
            case ENSEMBLE_CHAINS -> {
                double[] avg = new double[labels];
                for (int c = 0; c < chains.size(); c++) {
                    double[] mg = chainMarginals(row, c);
                    for (int j = 0; j < labels; j++) {
                        avg[j] += mg[j] / chains.size();
                    }
                }
                yield threshold(avg);
            }
            case LABEL_POWERSET -> {
                int best = 0;
                double bestS = Double.NEGATIVE_INFINITY;
                double[] ext = new double[row.length + 1];
                System.arraycopy(row, 0, ext, 0, row.length);
                ext[row.length] = 1;
                for (int c = 0; c < powersets.size(); c++) {
                    double s = 0;
                    for (int j = 0; j < ext.length; j++) {
                        s += powersetModels[c][j] * ext[j];
                    }
                    if (s > bestS) {
                        bestS = s;
                        best = c;
                    }
                }
                yield powersets.get(best).clone();
            }
            case ML_KNN -> knn(row);
        };
    }

    private static int[] threshold(double[] p) {
        int[] o = new int[p.length];
        for (int j = 0; j < p.length; j++) {
            o[j] = p[j] > 0.5 ? 1 : 0;
        }
        return o;
    }

    /** Greedy chain decoding; returns marginal probabilities in label order. */
    private double[] chainMarginals(double[] row, int c) {
        int[] order = orders.get(c);
        double[][] models = chains.get(c);
        boolean chained = method != Method.BINARY_RELEVANCE;
        double[] prev = new double[labels];
        double[] out = new double[labels];
        for (int a = 0; a < labels; a++) {
            double pr = prob(models[a], row, prev, chained ? a : 0);
            out[order[a]] = pr;
            prev[a] = pr > 0.5 ? 1 : 0;
        }
        return out;
    }

    private int[] beam(double[] row, int[] order, double[][] models) {
        List<double[]> beams = new ArrayList<>(); // prefix labels..., log-probability last
        beams.add(new double[] {0});
        for (int a = 0; a < labels; a++) {
            List<double[]> next = new ArrayList<>();
            for (double[] b : beams) {
                double pr = prob(models[a], row, b, a);
                for (int v = 0; v <= 1; v++) {
                    double[] nb = new double[a + 2];
                    System.arraycopy(b, 0, nb, 0, a);
                    nb[a] = v;
                    nb[a + 1] = b[a] + Math.log(Math.max(1e-12, v == 1 ? pr : 1 - pr));
                    next.add(nb);
                }
            }
            final int col = a + 1;
            next.sort((u, w) -> Double.compare(w[col], u[col]));
            beams = new ArrayList<>(next.subList(0, Math.min(beamWidth, next.size())));
        }
        double[] best = beams.get(0);
        int[] out = new int[labels];
        for (int a = 0; a < labels; a++) {
            out[order[a]] = (int) best[a];
        }
        return out;
    }

    private void trainPowerset(double[][] x, int[][] y) {
        Map<String, Integer> index = new HashMap<>();
        powersets = new ArrayList<>();
        int[] cls = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            String key = java.util.Arrays.toString(y[i]);
            Integer c = index.get(key);
            if (c == null) {
                c = powersets.size();
                index.put(key, c);
                powersets.add(y[i].clone());
            }
            cls[i] = c;
        }
        int p = x[0].length;
        double[][] f = new double[x.length][p + 1];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, f[i], 0, p);
            f[i][p] = 1;
        }
        powersetModels = new double[powersets.size()][];
        for (int c = 0; c < powersets.size(); c++) {
            double[] t = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                t[i] = cls[i] == c ? 1 : 0;
            }
            powersetModels[c] = Recalibrators.logistic(f, t, 1e-3);
        }
    }

    private void trainKnn(double[][] x, int[][] y, int neighbours) {
        k = neighbours;
        trainX = x;
        trainY = y;
        int n = x.length;
        prior = new double[labels];
        countGiven = new double[labels][2][k + 1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < labels; j++) {
                prior[j] += y[i][j];
            }
            int[] nb = neighbours(x[i], i);
            for (int j = 0; j < labels; j++) {
                int c = 0;
                for (int q : nb) {
                    c += y[q][j];
                }
                countGiven[j][y[i][j]][c]++;
            }
        }
        for (int j = 0; j < labels; j++) {
            prior[j] = (1 + prior[j]) / (2 + n);
            for (int h = 0; h <= 1; h++) {
                double tot = 0;
                for (double v : countGiven[j][h]) {
                    tot += v;
                }
                for (int c = 0; c <= k; c++) {
                    countGiven[j][h][c] = (1 + countGiven[j][h][c]) / (k + 1 + tot);
                }
            }
        }
    }

    private int[] neighbours(double[] row, int exclude) {
        int n = trainX.length;
        double[] best = new double[k];
        int[] idx = new int[k];
        java.util.Arrays.fill(best, Double.POSITIVE_INFINITY);
        for (int i = 0; i < n; i++) {
            if (i == exclude) {
                continue;
            }
            double d = 0;
            for (int j = 0; j < row.length; j++) {
                d += (row[j] - trainX[i][j]) * (row[j] - trainX[i][j]);
            }
            if (d < best[k - 1]) {
                int a = k - 1;
                while (a > 0 && best[a - 1] > d) {
                    best[a] = best[a - 1];
                    idx[a] = idx[a - 1];
                    a--;
                }
                best[a] = d;
                idx[a] = i;
            }
        }
        return idx;
    }

    private int[] knn(double[] row) {
        int[] nb = neighbours(row, -1);
        int[] out = new int[labels];
        for (int j = 0; j < labels; j++) {
            int c = 0;
            for (int q : nb) {
                c += trainY[q][j];
            }
            double p1 = prior[j] * countGiven[j][1][c];
            double p0 = (1 - prior[j]) * countGiven[j][0][c];
            out[j] = p1 > p0 ? 1 : 0;
        }
        return out;
    }

    /**
     * Hamming loss, subset accuracy and micro-F1.
     *
     * @param predicted predicted label sets
     * @param truth true label sets
     * @return {@code {Hamming loss, subset accuracy, micro-F1}}
     */
    public static double[] scores(int[][] predicted, int[][] truth) {
        int n = truth.length;
        int l = truth[0].length;
        int wrong = 0;
        int exact = 0;
        int tp = 0;
        int fp = 0;
        int fn = 0;
        for (int i = 0; i < n; i++) {
            boolean all = true;
            for (int j = 0; j < l; j++) {
                if (predicted[i][j] != truth[i][j]) {
                    wrong++;
                    all = false;
                }
                if (predicted[i][j] == 1 && truth[i][j] == 1) {
                    tp++;
                } else if (predicted[i][j] == 1) {
                    fp++;
                } else if (truth[i][j] == 1) {
                    fn++;
                }
            }
            if (all) {
                exact++;
            }
        }
        return new double[] {(double) wrong / (n * l), (double) exact / n,
            tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn)};
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }
}
