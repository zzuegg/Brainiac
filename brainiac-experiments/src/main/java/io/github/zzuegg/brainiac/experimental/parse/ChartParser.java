package io.github.zzuegg.brainiac.experimental.parse;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A span-based constituency parser (Stern, Andreas and Klein, ACL 2017,
 * <a href="https://arxiv.org/abs/1705.03919">arXiv:1705.03919</a>; Kitaev and Klein, ACL 2018,
 * <a href="https://arxiv.org/abs/1805.01052">arXiv:1805.01052</a>). The sentence
 * {@code [BOS, w_1 .. w_n, EOS]} is encoded by a BiLSTM or by a self-attentive encoder with
 * factored content and position attention; a span {@code (i, j)} between fenceposts is represented
 * by {@code [f_j - f_i; b_i - b_j]} (forward and backward halves) and an MLP gives a score per
 * label, the empty label {@code 0} scoring 0. The best binary tree (every span in it labelled,
 * possibly empty) is found by CKY. No grammar and no transition system are needed. Training:
 * <ul>
 *   <li>{@link Objective#MARGIN}: the structured hinge with the Hamming loss over labelled spans,
 *       by loss-augmented CKY;</li>
 *   <li>{@link Objective#TREE_CRF}: the tree-CRF likelihood, whose partition function is the
 *       inside algorithm and whose gradient is span-label marginals from inside-outside.</li>
 * </ul>
 * The dynamic programmes run in doubles; the network receives a surrogate loss
 * {@code sum(scores * (predicted - gold))} with the same gradient.
 */
public final class ChartParser {

    /** The training objective. */
    public enum Objective {
        /** Structured margin. */
        MARGIN,
        /** Tree-CRF likelihood. */
        TREE_CRF
    }

    /** The encoder. */
    public enum Encoder {
        /** A bidirectional LSTM. */
        BILSTM,
        /** The factored self-attentive encoder. */
        FACTORED_ATTENTION
    }

    private final int vocab;
    private final int labels;
    private final SentenceEncoder encoder;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final List<Var> params = new ArrayList<>();

    /**
     * Creates a parser.
     *
     * @param vocab the vocabulary (ids {@code vocab}, {@code vocab + 1} are BOS and EOS)
     * @param labels real phrase labels (output label {@code k + 1} is phrase {@code k})
     * @param encoder the encoder
     * @param embed the word embedding width (BiLSTM)
     * @param units units per direction (or per stream)
     * @param layers attention layers
     * @param hidden the span MLP width
     * @param seed the initialisation seed
     */
    public ChartParser(int vocab, int labels, Encoder encoder, int embed, int units, int layers, int hidden,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.vocab = vocab;
        this.labels = labels;
        this.encoder = new SentenceEncoder(encoder == Encoder.BILSTM ? SentenceEncoder.Kind.BILSTM
                : SentenceEncoder.Kind.FACTORED_ATTENTION, vocab + 2, embed, units, layers, r);
        int w = this.encoder.width();
        w1 = SentenceEncoder.matrix(r, w, hidden);
        b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        w2 = SentenceEncoder.matrix(r, hidden, labels);
        params.addAll(this.encoder.parameters());
        params.addAll(List.of(w1, b1, w2));
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    /**
     * The index of span {@code (i, j)}, {@code 0 <= i < j <= n}, in the score tensor.
     *
     * @param i the start fencepost
     * @param j the end fencepost
     * @param n the sentence length
     * @return the index
     */
    static int span(int i, int j, int n) {
        // spans ordered by start, then end
        return i * n - i * (i - 1) / 2 + (j - i - 1);
    }

    /**
     * Span scores {@code [batch, spans, labels + 1]} for equal-length sentences, column 0 (empty)
     * fixed at 0.
     *
     * @param words {@code [batch][n]} word ids
     * @return the scores
     */
    public Var spanScores(int[][] words) {
        int b = words.length;
        int n = words[0].length;
        int[][] x = new int[b][n + 2];
        for (int i = 0; i < b; i++) {
            x[i][0] = vocab;
            System.arraycopy(words[i], 0, x[i], 1, n);
            x[i][n + 1] = vocab + 1;
        }
        Var out = encoder.encode(x);
        int half = encoder.width() / 2;
        Var fwd = out.slice(2, 0, half).slice(1, 0, n + 1);
        Var bwd = out.slice(2, half, 2 * half).slice(1, 1, n + 2);
        int spans = n * (n + 1) / 2;
        float[] is = new float[spans];
        float[] js = new float[spans];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                is[span(i, j, n)] = i;
                js[span(i, j, n)] = j;
            }
        }
        Tensor ti = Tensor.of(Shape.of(spans), is);
        Tensor tj = Tensor.of(Shape.of(spans), js);
        Var rep = Var.concat(-1, fwd.gather(1, tj).sub(fwd.gather(1, ti)), bwd.gather(1, ti).sub(bwd.gather(1, tj)));
        Var s = rep.matmul(w1).add(b1).relu().matmul(w2);
        return Var.concat(-1, Var.of(Tensor.zeros(Shape.of(b, spans, 1))), s);
    }

    /**
     * The gold binarised tree as a label (0 = empty) for every span index, {@code -1} for spans
     * not in the tree. N-ary nodes are right-factored with empty-labelled intermediate spans.
     *
     * @param gold {@code {phrase, start, end}} constituents
     * @param n the length
     * @return labels per span index
     */
    static int[] goldTree(List<int[]> gold, int n) {
        int[][] label = new int[n + 1][n + 1];
        for (int[] row : label) {
            Arrays.fill(row, -1);
        }
        for (int[] g : gold) {
            label[g[1]][g[2]] = g[0] + 1;
        }
        int[] out = new int[n * (n + 1) / 2];
        Arrays.fill(out, -1);
        binarise(0, n, n, label, out);
        return out;
    }

    private static void binarise(int i, int j, int n, int[][] label, int[] out) {
        out[span(i, j, n)] = Math.max(0, label[i][j]);
        if (j - i == 1) {
            return;
        }
        int k = i + 1;
        for (int e = j - 1; e > i; e--) {
            if (label[i][e] >= 0) {
                k = e;
                break;
            }
        }
        binarise(i, k, n, label, out);
        binarise(k, j, n, label, out);
    }

    /**
     * CKY: the best binary tree under span-label scores.
     *
     * @param s {@code s[span][label]}
     * @param n the length
     * @return the label (0 = empty) of each span index in the tree, {@code -1} elsewhere
     */
    static int[] cky(double[][] s, int n) {
        double[] best = new double[s.length];
        int[] lab = new int[s.length];
        int[] split = new int[s.length];
        for (int len = 1; len <= n; len++) {
            for (int i = 0; i + len <= n; i++) {
                int j = i + len;
                int id = span(i, j, n);
                int l = 0;
                for (int k = 1; k < s[id].length; k++) {
                    if (s[id][k] > s[id][l]) {
                        l = k;
                    }
                }
                lab[id] = l;
                double inner = 0;
                if (len > 1) {
                    inner = Double.NEGATIVE_INFINITY;
                    for (int k = i + 1; k < j; k++) {
                        double v = best[span(i, k, n)] + best[span(k, j, n)];
                        if (v > inner) {
                            inner = v;
                            split[id] = k;
                        }
                    }
                }
                best[id] = s[id][l] + inner;
            }
        }
        int[] out = new int[s.length];
        Arrays.fill(out, -1);
        collect(0, n, n, lab, split, out);
        return out;
    }

    private static void collect(int i, int j, int n, int[] lab, int[] split, int[] out) {
        int id = span(i, j, n);
        out[id] = lab[id];
        if (j - i > 1) {
            collect(i, split[id], n, lab, split, out);
            collect(split[id], j, n, lab, split, out);
        }
    }

    /**
     * Span-label marginals of the tree CRF by inside-outside.
     *
     * @param s {@code s[span][label]}
     * @param n the length
     * @param logZ receives the log partition function at index 0
     * @return marginals {@code [span][label]}
     */
    static double[][] marginals(double[][] s, int n, double[] logZ) {
        int spans = s.length;
        double[] u = new double[spans];
        double[] beta = new double[spans];
        for (int id = 0; id < spans; id++) {
            u[id] = lse(s[id]);
        }
        for (int len = 1; len <= n; len++) {
            for (int i = 0; i + len <= n; i++) {
                int j = i + len;
                int id = span(i, j, n);
                double inner = 0;
                if (len > 1) {
                    double[] parts = new double[len - 1];
                    for (int k = i + 1; k < j; k++) {
                        parts[k - i - 1] = beta[span(i, k, n)] + beta[span(k, j, n)];
                    }
                    inner = lse(parts);
                }
                beta[id] = u[id] + inner;
            }
        }
        double[] alpha = new double[spans];
        Arrays.fill(alpha, Double.NEGATIVE_INFINITY);
        alpha[span(0, n, n)] = 0;
        for (int len = n; len >= 2; len--) {
            for (int i = 0; i + len <= n; i++) {
                int j = i + len;
                int id = span(i, j, n);
                double out = alpha[id] + u[id];
                for (int k = i + 1; k < j; k++) {
                    int left = span(i, k, n);
                    int right = span(k, j, n);
                    alpha[left] = logAdd(alpha[left], out + beta[right]);
                    alpha[right] = logAdd(alpha[right], out + beta[left]);
                }
            }
        }
        double z = beta[span(0, n, n)];
        logZ[0] = z;
        double[][] m = new double[spans][];
        for (int id = 0; id < spans; id++) {
            double p = Math.exp(alpha[id] + beta[id] - z);
            m[id] = new double[s[id].length];
            for (int l = 0; l < s[id].length; l++) {
                m[id][l] = p * Math.exp(s[id][l] - u[id]);
            }
        }
        return m;
    }

    private static double lse(double[] a) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : a) {
            max = Math.max(max, v);
        }
        double sum = 0;
        for (double v : a) {
            sum += Math.exp(v - max);
        }
        return max + Math.log(sum);
    }

    private static double logAdd(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) {
            return b;
        }
        double max = Math.max(a, b);
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    }

    /**
     * One training step's loss on equal-length sentences: returns the surrogate whose gradient is
     * the objective's, and writes the true mean loss into {@code value[0]}.
     *
     * @param words {@code [batch][n]} word ids
     * @param gold constituents per sentence
     * @param objective the objective
     * @param value receives the mean loss
     * @return the surrogate
     */
    public Var loss(int[][] words, List<List<int[]>> gold, Objective objective, double[] value) {
        int b = words.length;
        int n = words[0].length;
        int spans = n * (n + 1) / 2;
        int l = labels + 1;
        Var scores = spanScores(words);
        float[] all = scores.value().toArray();
        float[] coef = new float[b * spans * l];
        double total = 0;
        for (int i = 0; i < b; i++) {
            int[] g = goldTree(gold.get(i), n);
            double[][] s = new double[spans][l];
            double goldScore = 0;
            for (int id = 0; id < spans; id++) {
                for (int k = 0; k < l; k++) {
                    s[id][k] = all[(i * spans + id) * l + k];
                }
                if (g[id] >= 0) {
                    goldScore += s[id][g[id]];
                    coef[(i * spans + id) * l + g[id]] -= 1;
                }
            }
            if (objective == Objective.MARGIN) {
                for (int id = 0; id < spans; id++) {
                    int want = Math.max(0, g[id]);
                    for (int k = 0; k < l; k++) {
                        s[id][k] += k == want ? 0 : 1;
                    }
                }
                int[] p = cky(s, n);
                double predScore = 0;
                for (int id = 0; id < spans; id++) {
                    if (p[id] >= 0) {
                        predScore += s[id][p[id]];
                        coef[(i * spans + id) * l + p[id]] += 1;
                    }
                }
                total += Math.max(0, predScore - goldScore);
            } else {
                double[] z = new double[1];
                double[][] m = marginals(s, n, z);
                for (int id = 0; id < spans; id++) {
                    for (int k = 0; k < l; k++) {
                        coef[(i * spans + id) * l + k] += (float) m[id][k];
                    }
                }
                total += z[0] - goldScore;
            }
        }
        value[0] = total / b;
        return scores.mul(Tensor.of(Shape.of(b, spans, l), coef)).sum().mul(1f / b);
    }

    /**
     * Trains with Adam on batches of equal-length sentences.
     *
     * @param words sentences
     * @param gold constituents per sentence
     * @param objective the objective
     * @param steps optimiser steps
     * @param batch sentences per step
     * @param lr the learning rate
     * @param seed the sampling seed
     * @return the final mean loss
     */
    public double train(int[][] words, List<List<int[]>> gold, Objective objective, int steps, int batch, float lr,
            long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int[] lengths = Arrays.stream(words).mapToInt(w -> w.length).toArray();
        double[] value = new double[1];
        for (int s = 0; s < steps; s++) {
            int[] pick = SentenceEncoder.sameLengthBatch(lengths, batch, r);
            int[][] w = new int[batch][];
            List<List<int[]>> g = new ArrayList<>();
            for (int i = 0; i < batch; i++) {
                w[i] = words[pick[i]];
                g.add(gold.get(pick[i]));
            }
            Var loss = loss(w, g, objective, value);
            loss.backward();
            opt.step(params);
        }
        return value[0];
    }

    /**
     * Parses one sentence.
     *
     * @param words the word ids
     * @return labelled constituents {@code {phrase, start, end}} (empty-labelled spans dropped)
     */
    public List<int[]> parse(int[] words) {
        int n = words.length;
        int spans = n * (n + 1) / 2;
        int l = labels + 1;
        float[] all = Var.noGrad(() -> spanScores(new int[][] {words}).value().toArray());
        double[][] s = new double[spans][l];
        for (int id = 0; id < spans; id++) {
            for (int k = 0; k < l; k++) {
                s[id][k] = all[id * l + k];
            }
        }
        int[] tree = cky(s, n);
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                int lab = tree[span(i, j, n)];
                if (lab > 0) {
                    out.add(new int[] {lab - 1, i, j});
                }
            }
        }
        return out;
    }
}
