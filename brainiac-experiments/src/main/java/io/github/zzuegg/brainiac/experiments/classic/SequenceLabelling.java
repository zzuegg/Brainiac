package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.experimental.nn.LinearChainCrf;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Structured sequence labelling: hidden Markov models and a linear-chain CRF against neural
 * taggers.
 * <ul>
 *   <li>{@link #casino} and {@link #regimes}: the "dishonest casino" (a fair and a loaded die,
 *       discrete emissions) and a two-regime Gaussian series. Viterbi decoding with the true
 *       parameters and with Baum-Welch parameters learned from unlabelled sequences, against a
 *       GRU tagger trained with the true states; held-out log-likelihood per step of the HMMs
 *       against an i.i.d. baseline.</li>
 *   <li>{@link #chunking}: a toy chunking task with BIO tags of two entity types, where an
 *       {@code I-} tag may only follow a {@code B-} or {@code I-} tag of the same type and a
 *       span's type is only revealed by its last word. The same left-to-right GRU encoder
 *       trained with a per-token softmax and with a {@link LinearChainCrf} head, whose Viterbi
 *       decoding carries the type back through the span: span F1 and the rate of illegal
 *       transitions.</li>
 * </ul>
 */
public final class SequenceLabelling {

    private SequenceLabelling() {
    }

    /**
     * The dishonest casino: a fair die and a die that shows six half the time.
     *
     * @return the true model (state 0 fair, 1 loaded)
     */
    public static HiddenMarkovModel casinoModel() {
        double[][] emit = {{1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0, 1 / 6.0}, {0.1, 0.1, 0.1, 0.1, 0.1, 0.5}};
        return HiddenMarkovModel.discrete(new double[] {0.5, 0.5}, new double[][] {{0.95, 0.05}, {0.1, 0.9}}, emit);
    }

    /**
     * Two persistent regimes with different means.
     *
     * @return the true model
     */
    public static HiddenMarkovModel regimeModel() {
        return HiddenMarkovModel.gaussian(new double[] {0.5, 0.5}, new double[][] {{0.95, 0.05}, {0.05, 0.95}},
                new double[] {-1, 1}, new double[] {0.8, 0.8});
    }

    /**
     * The casino comparison.
     *
     * @param seed the seed
     * @return method to {decoding accuracy, held-out log-likelihood per step (NaN for the GRU)}
     */
    public static Map<String, double[]> casino(long seed) {
        HiddenMarkovModel truth = casinoModel();
        return compare(truth, HiddenMarkovModel.randomDiscrete(2, 6, seed), 6, seed);
    }

    /**
     * The regime-switching comparison.
     *
     * @param seed the seed
     * @return method to {decoding accuracy, held-out log-likelihood per step (NaN for the GRU)}
     */
    public static Map<String, double[]> regimes(long seed) {
        HiddenMarkovModel start = HiddenMarkovModel.gaussian(new double[] {0.5, 0.5},
                new double[][] {{0.8, 0.2}, {0.2, 0.8}}, new double[] {-0.3, 0.3}, new double[] {2, 2});
        return compare(regimeModel(), start, 0, seed);
    }

    private static Map<String, double[]> compare(HiddenMarkovModel truth, HiddenMarkovModel start, int alphabet,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int length = 300;
        List<double[][]> train = new ArrayList<>();
        List<double[][]> test = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            train.add(truth.sample(length, r));
        }
        for (int i = 0; i < 10; i++) {
            test.add(truth.sample(length, r));
        }
        List<double[]> unlabelled = new ArrayList<>();
        for (double[][] s : train) {
            unlabelled.add(s[0]);
        }
        HiddenMarkovModel learned = start.baumWelch(unlabelled, 40);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("HMM true params", new double[] {decodeAccuracy(truth, test, false), heldOut(truth, test)});
        out.put("HMM Baum-Welch", new double[] {decodeAccuracy(learned, test, true), heldOut(learned, test)});
        out.put("GRU tagger (supervised)", new double[] {gruAccuracy(train, test, alphabet, seed), Double.NaN});
        out.put("i.i.d. baseline", new double[] {Double.NaN, iidLogLikelihood(unlabelled, test, alphabet)});
        return out;
    }

    private static double decodeAccuracy(HiddenMarkovModel m, List<double[][]> test, boolean allowSwap) {
        int right = 0;
        int total = 0;
        int swapped = 0;
        for (double[][] s : test) {
            int[] path = m.viterbi(s[0]);
            for (int t = 0; t < path.length; t++) {
                right += path[t] == (int) s[1][t] ? 1 : 0;
                swapped += path[t] == 1 - (int) s[1][t] ? 1 : 0;
                total++;
            }
        }
        return (allowSwap ? Math.max(right, swapped) : right) / (double) total;
    }

    private static double heldOut(HiddenMarkovModel m, List<double[][]> test) {
        double ll = 0;
        int steps = 0;
        for (double[][] s : test) {
            ll += m.logLikelihood(s[0]);
            steps += s[0].length;
        }
        return ll / steps;
    }

    private static double iidLogLikelihood(List<double[]> train, List<double[][]> test, int alphabet) {
        double ll = 0;
        int steps = 0;
        if (alphabet > 0) {
            double[] p = new double[alphabet];
            int n = 0;
            for (double[] s : train) {
                for (double o : s) {
                    p[(int) o]++;
                    n++;
                }
            }
            for (double[][] s : test) {
                for (double o : s[0]) {
                    ll += Math.log(p[(int) o] / n);
                    steps++;
                }
            }
        } else {
            double m = 0;
            double sq = 0;
            int n = 0;
            for (double[] s : train) {
                for (double o : s) {
                    m += o;
                    sq += o * o;
                    n++;
                }
            }
            m /= n;
            double var = sq / n - m * m;
            for (double[][] s : test) {
                for (double o : s[0]) {
                    ll += -0.5 * Math.log(2 * Math.PI * var) - 0.5 * (o - m) * (o - m) / var;
                    steps++;
                }
            }
        }
        return ll / steps;
    }

    /** Inputs {@code [n, window, features]}: one-hot symbols, or the raw value. */
    private static Tensor inputs(List<double[]> windows, int alphabet) {
        int n = windows.size();
        int w = windows.getFirst().length;
        int f = Math.max(1, alphabet);
        float[] x = new float[n * w * f];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < w; t++) {
                double o = windows.get(i)[t];
                if (alphabet > 0) {
                    x[(i * w + t) * f + (int) o] = 1;
                } else {
                    x[(i * w + t) * f] = (float) o;
                }
            }
        }
        return Tensor.of(Shape.of(n, w, f), x);
    }

    /** A bidirectional GRU tagger trained on 50-step windows with the true states. */
    private static double gruAccuracy(List<double[][]> train, List<double[][]> test, int alphabet, long seed) {
        int w = 50;
        int f = Math.max(1, alphabet);
        Model net = Model.sequential().input(w, f).add(Recurrent.gru(16).returnSequences(true)
                .bidirectional(Recurrent.Merge.CONCAT)).dense(2).seed(seed).build();
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = 32;
        for (int step = 0; step < 150; step++) {
            List<double[]> xs = new ArrayList<>();
            float[] ys = new float[batch * w];
            for (int b = 0; b < batch; b++) {
                double[][] s = train.get(r.nextInt(train.size()));
                int o = r.nextInt(s[0].length - w + 1);
                xs.add(java.util.Arrays.copyOfRange(s[0], o, o + w));
                for (int t = 0; t < w; t++) {
                    ys[b * w + t] = (float) s[1][o + t];
                }
            }
            Var logits = net.forward(Var.of(inputs(xs, alphabet)), true).reshape(batch * w, 2);
            Functional.crossEntropy(logits, Tensor.of(Shape.of(batch * w), ys)).backward();
            opt.step(net.parameters());
        }
        int right = 0;
        int total = 0;
        for (double[][] s : test) {
            List<double[]> xs = new ArrayList<>();
            for (int o = 0; o + w <= s[0].length; o += w) {
                xs.add(java.util.Arrays.copyOfRange(s[0], o, o + w));
            }
            float[] p = net.forward(Var.of(inputs(xs, alphabet)), false).value().toArray();
            for (int i = 0; i < xs.size() * w; i++) {
                int pred = p[2 * i + 1] > p[2 * i] ? 1 : 0;
                right += pred == (int) s[1][i] ? 1 : 0;
                total++;
            }
        }
        return right / (double) total;
    }

    // ------------------------------------------------------------------ chunking with a CRF

    /** The BIO tags: O, B-A, I-A, B-B, I-B. */
    static final int TAGS = 5;

    /**
     * Generates chunking sentences: outside words are 10..19; an entity span is 2 to 5 words
     * from the shared set {3, 4, 8, 9} and ends with a type word (0..2 for type A, 5..7 for
     * type B), so the type of a span's first words is only revealed at its end.
     *
     * @param n sentences
     * @param length words per sentence
     * @param seed the seed
     * @return {words [n][length], tags [n][length]}
     */
    public static int[][][] chunks(int n, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] words = new int[n][length];
        int[][] tags = new int[n][length];
        int[] shared = {3, 4, 8, 9};
        for (int i = 0; i < n; i++) {
            int t = 0;
            while (t < length) {
                int span = 2 + r.nextInt(4);
                if (r.nextDouble() < 0.3 && t + span <= length) {
                    boolean a = r.nextBoolean();
                    for (int k = 0; k < span; k++, t++) {
                        words[i][t] = k == span - 1 ? (a ? r.nextInt(3) : 5 + r.nextInt(3))
                                : shared[r.nextInt(shared.length)];
                        tags[i][t] = k == 0 ? (a ? 1 : 3) : (a ? 2 : 4);
                    }
                } else {
                    words[i][t] = 10 + r.nextInt(10);
                    tags[i][t] = 0;
                    t++;
                }
            }
        }
        return new int[][][] {words, tags};
    }

    private static boolean illegal(int previous, int tag) {
        if (tag == 2) {
            return previous != 1 && previous != 2;
        }
        if (tag == 4) {
            return previous != 3 && previous != 4;
        }
        return false;
    }

    /** Exact-match span F1 over (start, end, type) spans read off BIO tags (illegal I- starts a span). */
    static double spanF1(int[][] gold, int[][] predicted) {
        int tp = 0;
        int np = 0;
        int ng = 0;
        for (int i = 0; i < gold.length; i++) {
            java.util.Set<String> g = spans(gold[i]);
            java.util.Set<String> p = spans(predicted[i]);
            ng += g.size();
            np += p.size();
            for (String s : p) {
                tp += g.contains(s) ? 1 : 0;
            }
        }
        double precision = np == 0 ? 0 : tp / (double) np;
        double recall = ng == 0 ? 0 : tp / (double) ng;
        return precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);
    }

    private static java.util.Set<String> spans(int[] tags) {
        java.util.Set<String> out = new java.util.HashSet<>();
        int t = 0;
        while (t < tags.length) {
            if (tags[t] == 0) {
                t++;
                continue;
            }
            int type = tags[t] <= 2 ? 0 : 1;
            int s = t;
            t++;
            while (t < tags.length && tags[t] == (type == 0 ? 2 : 4)) {
                t++;
            }
            out.add(s + ":" + t + ":" + type);
        }
        return out;
    }

    /**
     * The chunking comparison: the same encoder with a softmax head and with a CRF head.
     *
     * @param steps training steps per model
     * @param seed the seed
     * @return head to {span F1, illegal-transition rate per predicted token}
     */
    public static Map<String, double[]> chunking(int steps, long seed) {
        int length = 20;
        int[][][] train = chunks(400, length, seed);
        int[][][] test = chunks(200, length, seed + 1);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (boolean crf : new boolean[] {false, true}) {
            Model.Builder b = Model.sequential().input(length).embedding(20, 12)
                    .add(Recurrent.gru(24).returnSequences(true)).dense(TAGS);
            if (crf) {
                b.add(LinearChainCrf.spec(TAGS));
            }
            Model net = b.seed(seed).build();
            LinearChainCrf head = crf ? (LinearChainCrf) net.graph().getLast().layer() : null;
            Optimizer opt = Optimizer.adam(0.01f);
            SplittableRandom r = new SplittableRandom(seed);
            int batch = 32;
            for (int s = 0; s < steps; s++) {
                float[] x = new float[batch * length];
                float[] y = new float[batch * length];
                for (int i = 0; i < batch; i++) {
                    int j = r.nextInt(train[0].length);
                    for (int t = 0; t < length; t++) {
                        x[i * length + t] = train[0][j][t];
                        y[i * length + t] = train[1][j][t];
                    }
                }
                Var emissions = net.forward(Var.of(Tensor.of(Shape.of(batch, length), x)), true);
                Var loss = crf ? head.loss().compute(emissions, Tensor.of(Shape.of(batch, length), y))
                        : Functional.crossEntropy(emissions.reshape(batch * length, TAGS),
                                Tensor.of(Shape.of(batch * length), y));
                loss.backward();
                opt.step(net.parameters());
            }
            int n = test[0].length;
            float[] x = new float[n * length];
            for (int i = 0; i < n; i++) {
                for (int t = 0; t < length; t++) {
                    x[i * length + t] = test[0][i][t];
                }
            }
            Tensor emissions = net.forward(Var.of(Tensor.of(Shape.of(n, length), x)), false).value();
            int[][] predicted;
            if (crf) {
                predicted = head.decode(emissions);
            } else {
                float[] e = emissions.toArray();
                predicted = new int[n][length];
                for (int i = 0; i < n; i++) {
                    for (int t = 0; t < length; t++) {
                        int best = 0;
                        for (int k = 1; k < TAGS; k++) {
                            if (e[(i * length + t) * TAGS + k] > e[(i * length + t) * TAGS + best]) {
                                best = k;
                            }
                        }
                        predicted[i][t] = best;
                    }
                }
            }
            int bad = 0;
            for (int[] p : predicted) {
                for (int t = 0; t < length; t++) {
                    bad += illegal(t == 0 ? 0 : p[t - 1], p[t]) ? 1 : 0;
                }
            }
            out.put(crf ? "GRU-CRF" : "GRU-softmax", new double[] {spanF1(test[1], predicted),
                bad / (double) (n * length)});
        }
        return out;
    }
}
