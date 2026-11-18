package io.github.zzuegg.brainiac.experimental.parse;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The deep biaffine dependency parser (Dozat and Manning, ICLR 2017,
 * <a href="https://arxiv.org/abs/1611.01734">arXiv:1611.01734</a>). A BiLSTM reads
 * {@code [ROOT, w_1 .. w_n]}; separate ReLU MLPs project every position into a head and a
 * dependent vector, and every arc is scored by a biaffine product
 * {@code s(h -> d) = d^T U h + w^T h}; a second biaffine classifier
 * {@code d^T U_l h + W_l [d; h] + b_l} labels the arcs (with the gold head while training).
 * Training is a softmax over heads for each word. Decoding:
 * <ul>
 *   <li>{@link Decoder#GREEDY}: each word's best head (the head-selection fallback; may not be a
 *       tree);</li>
 *   <li>{@link Decoder#EISNER}: Eisner's O(n^3) dynamic programme for the best projective tree
 *       (Eisner, COLING 1996), with a single root child;</li>
 *   <li>{@link Decoder#CHU_LIU_EDMONDS}: the maximum spanning arborescence (Chu and Liu 1965,
 *       Edmonds 1967), non-projective, with a single root child.</li>
 * </ul>
 * Word id {@code vocab} is the ROOT token.
 */
public final class BiaffineParser {

    /** A decoding algorithm. */
    public enum Decoder {
        /** Best head per word. */
        GREEDY,
        /** Best projective tree. */
        EISNER,
        /** Best spanning arborescence. */
        CHU_LIU_EDMONDS
    }

    private static final double NEG = -1e30;

    private final int vocab;
    private final int labels;
    private final int arcDim;
    private final int labelDim;
    private final SentenceEncoder encoder;
    private final Var headW;
    private final Var depW;
    private final Var arcU;
    private final Var arcW;
    private final Var labelHeadW;
    private final Var labelDepW;
    private final Var labelU;
    private final Var labelW;
    private final Var labelB;
    private final List<Var> params = new ArrayList<>();

    /**
     * Creates a parser.
     *
     * @param vocab the vocabulary size (id {@code vocab} is ROOT)
     * @param labels dependency labels
     * @param embed the word embedding width
     * @param units LSTM units per direction
     * @param arcDim the arc MLP width
     * @param labelDim the label MLP width
     * @param seed the initialisation seed
     */
    public BiaffineParser(int vocab, int labels, int embed, int units, int arcDim, int labelDim, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.vocab = vocab;
        this.labels = labels;
        this.arcDim = arcDim;
        this.labelDim = labelDim;
        encoder = new SentenceEncoder(SentenceEncoder.Kind.BILSTM, vocab + 1, embed, units, 1, r);
        int w = encoder.width();
        headW = SentenceEncoder.matrix(r, w, arcDim);
        depW = SentenceEncoder.matrix(r, w, arcDim);
        arcU = Var.param(Tensor.zeros(Shape.of(arcDim, arcDim)));
        arcW = Var.param(Tensor.zeros(Shape.of(arcDim, 1)));
        labelHeadW = SentenceEncoder.matrix(r, w, labelDim);
        labelDepW = SentenceEncoder.matrix(r, w, labelDim);
        labelU = Var.param(Tensor.zeros(Shape.of(labelDim, labels * labelDim)));
        labelW = SentenceEncoder.matrix(r, 2 * labelDim, labels);
        labelB = Var.param(Tensor.zeros(Shape.of(labels)));
        params.addAll(encoder.parameters());
        params.addAll(List.of(headW, depW, arcU, arcW, labelHeadW, labelDepW, labelU, labelW, labelB));
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    private int[][] withRoot(int[][] words) {
        int[][] x = new int[words.length][];
        for (int i = 0; i < words.length; i++) {
            x[i] = new int[words[i].length + 1];
            x[i][0] = vocab;
            System.arraycopy(words[i], 0, x[i], 1, words[i].length);
        }
        return x;
    }

    /** Arc scores {@code [batch, n + 1 (dependent), n + 1 (head)]} and the encoder states. */
    private Var[] arcs(int[][] words) {
        Var x = encoder.encode(withRoot(words));
        Var h = x.matmul(headW).relu();
        Var d = x.matmul(depW).relu();
        Var hT = h.permute(0, 2, 1);
        Var s = d.matmul(arcU).matmul(hT).add(h.matmul(arcW).permute(0, 2, 1));
        return new Var[] {s, x};
    }

    /** Label scores {@code [batch * n, labels]} for the given heads. */
    private Var labelScores(Var x, int[][] heads) {
        int b = heads.length;
        int n = heads[0].length;
        Var lh = x.matmul(labelHeadW).relu().reshape(b * (n + 1), labelDim);
        Var ld = x.matmul(labelDepW).relu().slice(1, 1, n + 1).reshape(b * n, labelDim);
        float[] idx = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < n; j++) {
                idx[i * n + j] = i * (n + 1) + heads[i][j];
            }
        }
        Var hg = lh.gather(0, Tensor.of(Shape.of(b * n), idx));
        Var bil = ld.matmul(labelU).reshape(b * n, labels, labelDim).mul(hg.reshape(b * n, 1, labelDim)).sum(-1, false);
        return bil.add(Var.concat(-1, ld, hg).matmul(labelW)).add(labelB);
    }

    /**
     * The mean head and label cross-entropy of a batch of sentences of equal length.
     *
     * @param words {@code [batch][n]} word ids
     * @param heads {@code [batch][n]} 1-based heads, 0 for the root
     * @param rels {@code [batch][n]} label ids
     * @return the loss
     */
    public Var loss(int[][] words, int[][] heads, int[][] rels) {
        int b = words.length;
        int n = words[0].length;
        Var[] a = arcs(words);
        float[] h = new float[b * n];
        float[] l = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < n; j++) {
                h[i * n + j] = heads[i][j];
                l[i * n + j] = rels[i][j];
            }
        }
        Var arcLoss = Functional.crossEntropy(a[0].slice(1, 1, n + 1).reshape(b * n, n + 1),
                Tensor.of(Shape.of(b * n), h));
        Var labelLoss = Functional.crossEntropy(labelScores(a[1], heads), Tensor.of(Shape.of(b * n), l));
        return arcLoss.add(labelLoss);
    }

    /**
     * Trains with Adam on batches of equal-length sentences.
     *
     * @param words sentences
     * @param heads gold heads
     * @param rels gold labels
     * @param steps optimiser steps
     * @param batch sentences per step
     * @param lr the learning rate
     * @param seed the sampling seed
     * @return the final loss
     */
    public double train(int[][] words, int[][] heads, int[][] rels, int steps, int batch, float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int[] lengths = Arrays.stream(words).mapToInt(w -> w.length).toArray();
        double last = 0;
        for (int s = 0; s < steps; s++) {
            int[] pick = SentenceEncoder.sameLengthBatch(lengths, batch, r);
            int[][] w = new int[batch][];
            int[][] h = new int[batch][];
            int[][] l = new int[batch][];
            for (int i = 0; i < batch; i++) {
                w[i] = words[pick[i]];
                h[i] = heads[pick[i]];
                l[i] = rels[pick[i]];
            }
            Var loss = loss(w, h, l);
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Arc scores of one sentence.
     *
     * @param words the word ids
     * @return {@code scores[head][dependent]} over positions {@code 0..n} (0 is ROOT)
     */
    public double[][] arcScores(int[] words) {
        int n = words.length;
        float[] s = Var.noGrad(() -> arcs(new int[][] {words})[0].value().toArray());
        double[][] out = new double[n + 1][n + 1];
        for (int d = 0; d <= n; d++) {
            for (int h = 0; h <= n; h++) {
                out[h][d] = d == 0 || h == d ? NEG : s[d * (n + 1) + h];
            }
        }
        return out;
    }

    /**
     * Parses one sentence.
     *
     * @param words the word ids
     * @param decoder the decoder
     * @return {@code {heads, labels}}, heads 1-based with 0 for the root
     */
    public int[][] parse(int[] words, Decoder decoder) {
        int n = words.length;
        double[][] s = arcScores(words);
        int[] full = switch (decoder) {
            case GREEDY -> greedy(s);
            case EISNER -> eisner(s);
            case CHU_LIU_EDMONDS -> chuLiuEdmonds(s);
        };
        int[] heads = Arrays.copyOfRange(full, 1, n + 1);
        float[] ls = Var.noGrad(() -> {
            Var x = encoder.encode(withRoot(new int[][] {words}));
            return labelScores(x, new int[][] {heads}).value().toArray();
        });
        int[] rels = new int[n];
        for (int j = 0; j < n; j++) {
            for (int k = 1; k < labels; k++) {
                if (ls[j * labels + k] > ls[j * labels + rels[j]]) {
                    rels[j] = k;
                }
            }
        }
        return new int[][] {heads, rels};
    }

    /**
     * Head selection.
     *
     * @param s {@code s[head][dependent]}
     * @return heads per position (entry 0 is {@code -1})
     */
    public static int[] greedy(double[][] s) {
        int n = s.length;
        int[] head = new int[n];
        head[0] = -1;
        for (int d = 1; d < n; d++) {
            int best = 0;
            for (int h = 0; h < n; h++) {
                if (h != d && s[h][d] > s[best][d]) {
                    best = h;
                }
            }
            head[d] = best;
        }
        return head;
    }

    /**
     * Whether heads form a tree rooted at 0 with exactly one root child.
     *
     * @param heads 1-based heads of words {@code 1..n} (index 0 = word 1)
     * @return whether it is a well-formed tree
     */
    public static boolean isTree(int[] heads) {
        int n = heads.length;
        int roots = 0;
        for (int h : heads) {
            roots += h == 0 ? 1 : 0;
        }
        if (roots != 1) {
            return false;
        }
        for (int i = 1; i <= n; i++) {
            int v = i;
            int steps = 0;
            while (v != 0 && steps <= n) {
                v = heads[v - 1];
                steps++;
            }
            if (v != 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * Eisner's algorithm with a single root child.
     *
     * @param s {@code s[head][dependent]} over positions {@code 0..n}
     * @return heads per position (entry 0 is {@code -1})
     */
    public static int[] eisner(double[][] s) {
        int n = s.length - 1;
        // tables over positions 0..n; dir 0: head on the right (left arc), dir 1: head on the left
        double[][][] c = new double[n + 1][n + 1][2];
        double[][][] in = new double[n + 1][n + 1][2];
        int[][][] cb = new int[n + 1][n + 1][2];
        int[][][] ib = new int[n + 1][n + 1][2];
        for (int k = 1; k <= n; k++) {
            for (int a = 0; a + k <= n; a++) {
                int b = a + k;
                double best = NEG;
                int arg = a;
                for (int r = a; r < b; r++) {
                    double v = c[a][r][1] + c[r + 1][b][0];
                    if (v > best) {
                        best = v;
                        arg = r;
                    }
                }
                in[a][b][0] = best + s[b][a];
                in[a][b][1] = best + s[a][b];
                ib[a][b][0] = arg;
                ib[a][b][1] = arg;
                best = NEG;
                for (int r = a; r < b; r++) {
                    double v = c[a][r][0] + in[r][b][0];
                    if (v > best) {
                        best = v;
                        arg = r;
                    }
                }
                c[a][b][0] = best;
                cb[a][b][0] = arg;
                best = NEG;
                for (int r = a + 1; r <= b; r++) {
                    double v = in[a][r][1] + c[r][b][1];
                    if (v > best) {
                        best = v;
                        arg = r;
                    }
                }
                c[a][b][1] = best;
                cb[a][b][1] = arg;
            }
        }
        int[] head = new int[n + 1];
        head[0] = -1;
        double best = NEG;
        int root = 1;
        for (int k = 1; k <= n; k++) {
            double v = (k > 1 ? c[1][k][0] : 0) + s[0][k] + c[k][n][1];
            if (v > best) {
                best = v;
                root = k;
            }
        }
        head[root] = 0;
        if (root > 1) {
            backComplete(1, root, 0, c, in, cb, ib, head);
        }
        backComplete(root, n, 1, c, in, cb, ib, head);
        return head;
    }

    private static void backComplete(int a, int b, int dir, double[][][] c, double[][][] in, int[][][] cb,
            int[][][] ib, int[] head) {
        if (a == b) {
            return;
        }
        int r = cb[a][b][dir];
        if (dir == 0) {
            backComplete(a, r, 0, c, in, cb, ib, head);
            backIncomplete(r, b, 0, c, in, cb, ib, head);
        } else {
            backIncomplete(a, r, 1, c, in, cb, ib, head);
            backComplete(r, b, 1, c, in, cb, ib, head);
        }
    }

    private static void backIncomplete(int a, int b, int dir, double[][][] c, double[][][] in, int[][][] cb,
            int[][][] ib, int[] head) {
        if (dir == 0) {
            head[a] = b;
        } else {
            head[b] = a;
        }
        int r = ib[a][b][dir];
        backComplete(a, r, 1, c, in, cb, ib, head);
        backComplete(r + 1, b, 0, c, in, cb, ib, head);
    }

    /**
     * The maximum spanning arborescence rooted at 0 with a single root child.
     *
     * @param s {@code s[head][dependent]} over positions {@code 0..n}
     * @return heads per position (entry 0 is {@code -1})
     */
    public static int[] chuLiuEdmonds(double[][] s) {
        int[] free = cle(s);
        int roots = 0;
        for (int d = 1; d < s.length; d++) {
            roots += free[d] == 0 ? 1 : 0;
        }
        if (roots == 1) {
            return free;
        }
        int[] best = null;
        double bestScore = NEG;
        for (int r = 1; r < s.length; r++) {
            double[][] t = new double[s.length][];
            for (int i = 0; i < s.length; i++) {
                t[i] = s[i].clone();
            }
            for (int d = 1; d < s.length; d++) {
                if (d != r) {
                    t[0][d] = NEG;
                }
            }
            int[] h = cle(t);
            double score = 0;
            for (int d = 1; d < s.length; d++) {
                score += s[h[d]][d];
            }
            if (best == null || score > bestScore) {
                best = h;
                bestScore = score;
            }
        }
        return best;
    }

    private static int[] cle(double[][] s) {
        int n = s.length;
        int[] head = new int[n];
        head[0] = -1;
        for (int d = 1; d < n; d++) {
            int best = 0;
            for (int h = 0; h < n; h++) {
                if (h != d && s[h][d] > s[best][d]) {
                    best = h;
                }
            }
            head[d] = best;
        }
        int[] cycle = findCycle(head);
        if (cycle == null) {
            return head;
        }
        boolean[] inCycle = new boolean[n];
        for (int v : cycle) {
            inCycle[v] = true;
        }
        int m = n - cycle.length + 1;
        int[] map = new int[n];
        int[] back = new int[m];
        int next = 0;
        for (int v = 0; v < n; v++) {
            if (!inCycle[v]) {
                map[v] = next;
                back[next++] = v;
            }
        }
        int cnode = m - 1;
        for (int v : cycle) {
            map[v] = cnode;
        }
        double[][] t = new double[m][m];
        for (double[] row : t) {
            Arrays.fill(row, NEG);
        }
        int[] enter = new int[m];
        int[] leave = new int[m];
        for (int u = 0; u < n; u++) {
            for (int v = 1; v < n; v++) {
                if (u == v || s[u][v] <= NEG) {
                    continue;
                }
                if (!inCycle[u] && !inCycle[v]) {
                    t[map[u]][map[v]] = s[u][v];
                } else if (!inCycle[u]) {
                    double val = s[u][v] - s[head[v]][v];
                    if (val > t[map[u]][cnode]) {
                        t[map[u]][cnode] = val;
                        enter[map[u]] = v;
                    }
                } else if (!inCycle[v]) {
                    if (s[u][v] > t[cnode][map[v]]) {
                        t[cnode][map[v]] = s[u][v];
                        leave[map[v]] = u;
                    }
                }
            }
        }
        int[] sub = cle(t);
        int[] out = head.clone();
        for (int v = 1; v < n; v++) {
            if (!inCycle[v]) {
                int h = sub[map[v]];
                out[v] = h == cnode ? leave[map[v]] : back[h];
            }
        }
        int h = sub[cnode];
        out[enter[h]] = back[h];
        return out;
    }

    private static int[] findCycle(int[] head) {
        int n = head.length;
        int[] color = new int[n];
        for (int start = 1; start < n; start++) {
            if (color[start] != 0) {
                continue;
            }
            int v = start;
            while (v > 0 && color[v] == 0) {
                color[v] = start;
                v = head[v];
            }
            if (v > 0 && color[v] == start) {
                List<Integer> c = new ArrayList<>();
                int u = v;
                do {
                    c.add(u);
                    u = head[u];
                } while (u != v);
                return c.stream().mapToInt(Integer::intValue).toArray();
            }
        }
        return null;
    }

    /** @return the arc MLP width */
    public int arcDim() {
        return arcDim;
    }
}
