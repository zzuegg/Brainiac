package io.github.zzuegg.brainiac.experimental.bio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.BioSequences;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * RNA secondary structure prediction for pseudoknot-free (nested) structures.
 *
 * <ul>
 *   <li>{@link #nussinov}: maximises the number of canonical base pairs (Nussinov and Jacobson,
 *       PNAS 1980).</li>
 *   <li>A loop-decomposition model in the Zuker style: every structure decomposes uniquely into
 *       hairpins, stacks, bulges and interior loops (up to {@link #MAX_LOOP} unpaired bases),
 *       multiloops (linear {@code a + b unpaired + c branches}) and the exterior loop. Each loop
 *       contributes a weighted sum of features (pair types, 6 x 6 stacking, hairpin/bulge/
 *       interior length bins, multiloop terms, unpaired base identities). The grammar is built
 *       once per sequence as a hypergraph, and every algorithm runs on it:
 *       {@link #fold} with {@link #energyWeights()} is the minimum-free-energy fold
 *       (SIMPLE_ZUKER, Zuker and Stiegler 1981); {@link #pairProbabilities} is McCaskill's
 *       inside-outside partition function (Biopolymers 1990) and {@link #meaFold} decodes the
 *       maximum-expected-accuracy structure from it; {@link #trainCrf} learns the weights
 *       discriminatively as a conditional random field, whose gradient is the observed minus
 *       the expected feature counts (CONTRAfold, Do, Woods and Batzoglou 2006).</li>
 *   <li>{@link PairCnn}: an unconstrained learner that predicts the pairing matrix from
 *       windowed one-hot embeddings (symmetric dot-product scores), decoded greedily; the
 *       baseline that may emit invalid structures.</li>
 * </ul>
 *
 * <p>Bases are {@code ACGU} = 0..3; a structure is the partner array ({@code -1} unpaired).
 */
public final class RnaFolding {

    /** The largest number of unpaired bases in a bulge or interior loop. */
    public static final int MAX_LOOP = 16;
    /** The smallest hairpin loop. */
    public static final int MIN_HAIRPIN = 3;

    // feature layout
    private static final int F_PAIR = 0;
    private static final int F_STACK = 6;
    private static final int F_HAIRPIN = 42;
    private static final int F_BULGE = 51;
    private static final int F_INTERIOR = 56;
    private static final int F_MULTI_A = 63;
    private static final int F_MULTI_B = 64;
    private static final int F_MULTI_C = 65;
    private static final int F_EXT_UNPAIRED = 66;
    private static final int F_EXT_BRANCH = 67;
    private static final int F_LOOP_BASE = 68;
    /** The number of model features. */
    public static final int FEATURES = 72;

    private RnaFolding() {
    }

    private static int pairType(int x, int y) {
        if (x == 0 && y == 3) {
            return 0;
        }
        if (x == 1 && y == 2) {
            return 1;
        }
        if (x == 2 && y == 1) {
            return 2;
        }
        if (x == 3 && y == 0) {
            return 3;
        }
        if (x == 2 && y == 3) {
            return 4;
        }
        if (x == 3 && y == 2) {
            return 5;
        }
        return -1;
    }

    /**
     * Nussinov's base-pair maximisation (canonical pairs, hairpins of at least
     * {@link #MIN_HAIRPIN}).
     *
     * @param seq the sequence
     * @return the partner array
     */
    public static int[] nussinov(int[] seq) {
        int n = seq.length;
        int[][] f = new int[n + 1][n + 1];
        for (int d = MIN_HAIRPIN + 1; d < n; d++) {
            for (int i = 0; i + d < n; i++) {
                int j = i + d;
                int best = f[i][j - 1];
                for (int k = i; k < j - MIN_HAIRPIN; k++) {
                    if (pairType(seq[k], seq[j]) >= 0) {
                        int v = (k > i ? f[i][k - 1] : 0) + f[k + 1][j - 1] + 1;
                        best = Math.max(best, v);
                    }
                }
                f[i][j] = best;
            }
        }
        int[] pairs = new int[n];
        Arrays.fill(pairs, -1);
        traceNussinov(seq, f, 0, n - 1, pairs);
        return pairs;
    }

    private static void traceNussinov(int[] seq, int[][] f, int i, int j, int[] pairs) {
        while (j - i > MIN_HAIRPIN) {
            if (f[i][j] == f[i][j - 1]) {
                j--;
                continue;
            }
            for (int k = i; k < j - MIN_HAIRPIN; k++) {
                if (pairType(seq[k], seq[j]) >= 0) {
                    int v = (k > i ? f[i][k - 1] : 0) + f[k + 1][j - 1] + 1;
                    if (v == f[i][j]) {
                        pairs[k] = j;
                        pairs[j] = k;
                        if (k > i) {
                            traceNussinov(seq, f, i, k - 1, pairs);
                        }
                        traceNussinov(seq, f, k + 1, j - 1, pairs);
                        return;
                    }
                }
            }
            return;
        }
    }

    /**
     * Hand-set weights of a simplified nearest-neighbour energy model, as scores
     * {@code -energy / kT} (kT = 0.6 kcal/mol): stacking from pair strengths (GC 3.0, AU 2.0,
     * GU 1.0 kcal/mol averaged over the two pairs), hairpins 5.4-6.5, bulges 3.8 and up,
     * interior loops {@code 1 + 0.5 size}, multiloops {@code a = 3.4, b = 0, c = 0.4} and a
     * terminal AU/GU penalty of 0.5.
     *
     * @return the weights
     */
    public static double[] energyWeights() {
        double kt = 0.6;
        double[] w = new double[FEATURES];
        double[] strength = {2.0, 3.0, 3.0, 2.0, 1.0, 1.0};
        for (int t = 0; t < 6; t++) {
            w[F_PAIR + t] = (t == 0 || t == 3 || t >= 4) ? -0.25 / kt : 0;
            for (int u = 0; u < 6; u++) {
                w[F_STACK + 6 * t + u] = (strength[t] + strength[u]) / 2 / kt;
            }
        }
        double[] hairpin = {5.4, 5.6, 5.7, 5.4, 6.0, 6.1, 6.2, 6.3, 6.5};
        for (int b = 0; b < 9; b++) {
            w[F_HAIRPIN + b] = -hairpin[b] / kt;
        }
        double[] bulge = {3.8, 2.8, 3.2, 3.6, 4.0};
        for (int b = 0; b < 5; b++) {
            w[F_BULGE + b] = -bulge[b] / kt;
        }
        for (int b = 0; b < 7; b++) {
            w[F_INTERIOR + b] = -(1.0 + 0.5 * (b + 2)) / kt;
        }
        w[F_MULTI_A] = -3.4 / kt;
        w[F_MULTI_B] = 0;
        w[F_MULTI_C] = -0.4 / kt;
        return w;
    }

    /** The loop-decomposition grammar of one sequence as a hypergraph. */
    private static final class Grammar {
        final int n;
        final int[] seq;
        int cells;
        final int[] pCell;
        final int[] mCell;
        final int[] m2Cell;
        final int[] eCell;
        // edges
        int edges;
        int[] head = new int[1024];
        int[] tail1 = new int[1024];
        int[] tail2 = new int[1024];
        int[] featStart = new int[1025];
        int[] featIdx = new int[4096];
        double[] featCnt = new double[4096];
        int[] pairI = new int[1024];
        int[] pairJ = new int[1024];
        int[] spans = new int[4096];
        int nFeat;
        // unpaired intervals per edge: up to two [a, b] (empty if a > b)
        int[] cellFirstEdge;
        int[] cellLastEdge;
        final int[][] baseCount;

        Grammar(int[] seq) {
            this.seq = seq;
            this.n = seq.length;
            pCell = new int[n * n];
            mCell = new int[n * n];
            m2Cell = new int[n * n];
            eCell = new int[n + 1];
            Arrays.fill(pCell, -1);
            Arrays.fill(mCell, -1);
            Arrays.fill(m2Cell, -1);
            baseCount = new int[n + 1][4];
            for (int i = 0; i < n; i++) {
                baseCount[i + 1] = baseCount[i].clone();
                baseCount[i + 1][seq[i]]++;
            }
            build();
        }

        private void grow() {
            int cap = head.length * 2;
            head = Arrays.copyOf(head, cap);
            tail1 = Arrays.copyOf(tail1, cap);
            tail2 = Arrays.copyOf(tail2, cap);
            featStart = Arrays.copyOf(featStart, cap + 1);
            pairI = Arrays.copyOf(pairI, cap);
            pairJ = Arrays.copyOf(pairJ, cap);
            spans = Arrays.copyOf(spans, cap * 4);
        }

        private int curHead;
        private final List<int[]> ranges = new ArrayList<>();

        private void beginCell() {
            curHead = cells++;
            ranges.add(new int[] {edges, edges});
        }

        private void endCell() {
            ranges.get(ranges.size() - 1)[1] = edges;
        }

        private void feat(int idx, double cnt) {
            if (cnt == 0) {
                return;
            }
            if (nFeat == featIdx.length) {
                featIdx = Arrays.copyOf(featIdx, nFeat * 2);
                featCnt = Arrays.copyOf(featCnt, nFeat * 2);
            }
            featIdx[nFeat] = idx;
            featCnt[nFeat++] = cnt;
        }

        private void loopBases(int a, int b) {
            if (a > b) {
                return;
            }
            for (int x = 0; x < 4; x++) {
                feat(F_LOOP_BASE + x, baseCount[b + 1][x] - baseCount[a][x]);
            }
        }

        /** Starts an edge into the current cell; features follow; then {@link #close}. */
        private void edge(int t1, int t2, int pi, int pj, int u1a, int u1b, int u2a, int u2b) {
            if (edges == head.length) {
                grow();
            }
            head[edges] = curHead;
            tail1[edges] = t1;
            tail2[edges] = t2;
            pairI[edges] = pi;
            pairJ[edges] = pj;
            spans[4 * edges] = u1a;
            spans[4 * edges + 1] = u1b;
            spans[4 * edges + 2] = u2a;
            spans[4 * edges + 3] = u2b;
            featStart[edges] = nFeat;
        }

        private void close() {
            edges++;
            featStart[edges] = nFeat;
        }

        private void build() {
            for (int d = MIN_HAIRPIN + 1; d < n; d++) {
                for (int i = 0; i + d < n; i++) {
                    int j = i + d;
                    int t = pairType(seq[i], seq[j]);
                    if (t >= 0) {
                        beginCell();
                        pCell[i * n + j] = curHead;
                        int len = j - i - 1;
                        // hairpin
                        edge(-1, -1, i, j, i + 1, j - 1, 1, 0);
                        feat(F_PAIR + t, 1);
                        feat(F_HAIRPIN + Math.min(len - 3, 8), 1);
                        loopBases(i + 1, j - 1);
                        close();
                        // stacks, bulges and interior loops
                        for (int k = i + 1; k < j && k - i - 1 <= MAX_LOOP; k++) {
                            for (int l = j - 1; l > k + MIN_HAIRPIN && (k - i - 1) + (j - l - 1) <= MAX_LOOP; l--) {
                                int inner = pCell[k * n + l];
                                if (inner < 0) {
                                    continue;
                                }
                                int left = k - i - 1;
                                int right = j - l - 1;
                                edge(inner, -1, i, j, i + 1, k - 1, l + 1, j - 1);
                                feat(F_PAIR + t, 1);
                                if (left == 0 && right == 0) {
                                    feat(F_STACK + 6 * t + pairType(seq[k], seq[l]), 1);
                                } else if (left == 0 || right == 0) {
                                    feat(F_BULGE + Math.min(left + right - 1, 4), 1);
                                } else {
                                    feat(F_INTERIOR + Math.min(left + right - 2, 6), 1);
                                }
                                loopBases(i + 1, k - 1);
                                loopBases(l + 1, j - 1);
                                close();
                            }
                        }
                        // multiloop closed by (i, j)
                        if (j - i - 1 >= 2 * (MIN_HAIRPIN + 2)) {
                            int inner = m2Cell[(i + 1) * n + (j - 1)];
                            if (inner >= 0) {
                                edge(inner, -1, i, j, 1, 0, 1, 0);
                                feat(F_PAIR + t, 1);
                                feat(F_MULTI_A, 1);
                                feat(F_MULTI_C, 1);
                                close();
                            }
                        }
                        endCell();
                    }
                    // M[i][j]: at least one branch
                    beginCell();
                    boolean any = false;
                    int next = i + 1 < n ? mCell[(i + 1) * n + j] : -1;
                    if (i + 1 <= j && next >= 0) {
                        edge(next, -1, -1, -1, i, i, 1, 0);
                        feat(F_MULTI_B, 1);
                        close();
                        any = true;
                    }
                    for (int k = i + MIN_HAIRPIN + 1; k <= j; k++) {
                        int p = pCell[i * n + k];
                        if (p < 0) {
                            continue;
                        }
                        edge(p, -1, -1, -1, k + 1, j, 1, 0);
                        feat(F_MULTI_C, 1);
                        feat(F_MULTI_B, j - k);
                        close();
                        any = true;
                        if (k + 1 < j) {
                            int rest = mCell[(k + 1) * n + j];
                            if (rest >= 0) {
                                edge(p, rest, -1, -1, 1, 0, 1, 0);
                                feat(F_MULTI_C, 1);
                                close();
                            }
                        }
                    }
                    endCell();
                    if (any) {
                        mCell[i * n + j] = curHead;
                    }
                    // M2[i][j]: at least two branches
                    beginCell();
                    boolean any2 = false;
                    int next2 = i + 1 < n ? m2Cell[(i + 1) * n + j] : -1;
                    if (next2 >= 0) {
                        edge(next2, -1, -1, -1, i, i, 1, 0);
                        feat(F_MULTI_B, 1);
                        close();
                        any2 = true;
                    }
                    for (int k = i + MIN_HAIRPIN + 1; k + 1 < j; k++) {
                        int p = pCell[i * n + k];
                        int rest = mCell[(k + 1) * n + j];
                        if (p < 0 || rest < 0) {
                            continue;
                        }
                        edge(p, rest, -1, -1, 1, 0, 1, 0);
                        feat(F_MULTI_C, 1);
                        close();
                        any2 = true;
                    }
                    endCell();
                    if (any2) {
                        m2Cell[i * n + j] = curHead;
                    }
                }
            }
            // exterior loop: E[j] covers positions 0..j-1
            beginCell();
            eCell[0] = curHead;
            edge(-1, -1, -1, -1, 1, 0, 1, 0);
            close();
            endCell();
            for (int j = 1; j <= n; j++) {
                beginCell();
                eCell[j] = curHead;
                edge(eCell[j - 1], -1, -1, -1, j - 1, j - 1, 1, 0);
                feat(F_EXT_UNPAIRED, 1);
                close();
                for (int i = 0; i < j; i++) {
                    int p = pCell[i * n + (j - 1)];
                    if (p >= 0) {
                        edge(eCell[i], p, -1, -1, 1, 0, 1, 0);
                        feat(F_EXT_BRANCH, 1);
                        close();
                    }
                }
                endCell();
            }
            cellFirstEdge = new int[cells];
            cellLastEdge = new int[cells];
            for (int c = 0; c < cells; c++) {
                cellFirstEdge[c] = ranges.get(c)[0];
                cellLastEdge[c] = ranges.get(c)[1];
            }
        }

        double edgeScore(int e, double[] w) {
            double s = 0;
            for (int f = featStart[e]; f < featStart[e + 1]; f++) {
                s += w[featIdx[f]] * featCnt[f];
            }
            return s;
        }

        boolean consistent(int e, int[] truth, int[] pairedPrefix) {
            if (pairI[e] >= 0 && truth[pairI[e]] != pairJ[e]) {
                return false;
            }
            for (int s = 0; s < 2; s++) {
                int a = spans[4 * e + 2 * s];
                int b = spans[4 * e + 2 * s + 1];
                if (a <= b && pairedPrefix[b + 1] - pairedPrefix[a] > 0) {
                    return false;
                }
            }
            return true;
        }

        /** Inside values in log space (or max); {@code arg} receives the best edge per cell. */
        double[] inside(double[] w, boolean[] allowed, boolean viterbi, int[] arg) {
            double[] v = new double[cells];
            for (int c = 0; c < cells; c++) {
                double best = Double.NEGATIVE_INFINITY;
                double sum = 0;
                int bestEdge = -1;
                double[] vals = new double[cellLastEdge[c] - cellFirstEdge[c]];
                for (int e = cellFirstEdge[c]; e < cellLastEdge[c]; e++) {
                    double x = Double.NEGATIVE_INFINITY;
                    if (allowed == null || allowed[e]) {
                        x = edgeScore(e, w) + (tail1[e] >= 0 ? v[tail1[e]] : 0) + (tail2[e] >= 0 ? v[tail2[e]] : 0);
                    }
                    vals[e - cellFirstEdge[c]] = x;
                    if (x > best) {
                        best = x;
                        bestEdge = e;
                    }
                }
                if (viterbi || best == Double.NEGATIVE_INFINITY) {
                    v[c] = best;
                } else {
                    for (double x : vals) {
                        sum += Math.exp(x - best);
                    }
                    v[c] = best + Math.log(sum);
                }
                if (arg != null) {
                    arg[c] = bestEdge;
                }
            }
            return v;
        }

        /** Edge posteriors by the outside pass. */
        double[] posteriors(double[] w, boolean[] allowed, double[] in) {
            double[] out = new double[cells];
            Arrays.fill(out, Double.NEGATIVE_INFINITY);
            int root = eCell[n];
            out[root] = 0;
            double logZ = in[root];
            double[] post = new double[edges];
            for (int c = cells - 1; c >= 0; c--) {
                if (out[c] == Double.NEGATIVE_INFINITY) {
                    continue;
                }
                for (int e = cellFirstEdge[c]; e < cellLastEdge[c]; e++) {
                    if (allowed != null && !allowed[e]) {
                        continue;
                    }
                    double t1 = tail1[e] >= 0 ? in[tail1[e]] : 0;
                    double t2 = tail2[e] >= 0 ? in[tail2[e]] : 0;
                    double s = edgeScore(e, w) + t1 + t2;
                    double logPost = out[c] + s - logZ;
                    post[e] = Math.exp(logPost);
                    if (tail1[e] >= 0) {
                        out[tail1[e]] = lse(out[tail1[e]], out[c] + s - t1);
                    }
                    if (tail2[e] >= 0) {
                        out[tail2[e]] = lse(out[tail2[e]], out[c] + s - t2);
                    }
                }
            }
            return post;
        }

        void collect(int c, int[] arg, int[] pairs, double[] feats) {
            int e = arg[c];
            if (e < 0) {
                return;
            }
            if (pairI[e] >= 0) {
                pairs[pairI[e]] = pairJ[e];
                pairs[pairJ[e]] = pairI[e];
            }
            if (feats != null) {
                for (int f = featStart[e]; f < featStart[e + 1]; f++) {
                    feats[featIdx[f]] += featCnt[f];
                }
            }
            if (tail1[e] >= 0) {
                collect(tail1[e], arg, pairs, feats);
            }
            if (tail2[e] >= 0) {
                collect(tail2[e], arg, pairs, feats);
            }
        }
    }

    private static double lse(double a, double b) {
        if (a == Double.NEGATIVE_INFINITY) {
            return b;
        }
        if (b == Double.NEGATIVE_INFINITY) {
            return a;
        }
        return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
    }

    /**
     * The best-scoring structure under weights: with {@link #energyWeights()} this is the
     * minimum-free-energy fold.
     *
     * @param seq the sequence
     * @param weights the feature weights
     * @return the partner array
     */
    public static int[] fold(int[] seq, double[] weights) {
        Grammar g = new Grammar(seq);
        int[] arg = new int[g.cells];
        g.inside(weights, null, true, arg);
        int[] pairs = new int[seq.length];
        Arrays.fill(pairs, -1);
        g.collect(g.eCell[seq.length], arg, pairs, null);
        return pairs;
    }

    /**
     * McCaskill's base-pair probabilities under the Boltzmann distribution of the weights.
     *
     * @param seq the sequence
     * @param weights the feature weights
     * @return {@code [i][j]} pair probabilities (symmetric)
     */
    public static double[][] pairProbabilities(int[] seq, double[] weights) {
        Grammar g = new Grammar(seq);
        double[] in = g.inside(weights, null, false, null);
        double[] post = g.posteriors(weights, null, in);
        int n = seq.length;
        double[][] p = new double[n][n];
        for (int e = 0; e < g.edges; e++) {
            if (g.pairI[e] >= 0) {
                p[g.pairI[e]][g.pairJ[e]] += post[e];
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                p[j][i] = p[i][j];
            }
        }
        return p;
    }

    /**
     * Maximum-expected-accuracy decoding: the nested structure maximising
     * {@code sum_pairs 2 gamma P_ij + sum_unpaired q_i}, with {@code q_i = 1 - sum_j P_ij}.
     *
     * @param probs pair probabilities
     * @param gamma the pair weight (higher favours sensitivity)
     * @return the partner array
     */
    public static int[] meaFold(double[][] probs, double gamma) {
        int n = probs.length;
        double[] q = new double[n];
        for (int i = 0; i < n; i++) {
            q[i] = 1;
            for (int j = 0; j < n; j++) {
                q[i] -= probs[i][j];
            }
        }
        double[][] f = new double[n + 1][n + 1];
        for (int d = 0; d < n; d++) {
            for (int i = 0; i + d < n; i++) {
                int j = i + d;
                double best = (j > i ? f[i][j - 1] : 0) + q[j];
                for (int k = i; k < j - MIN_HAIRPIN; k++) {
                    if (probs[k][j] > 0) {
                        double v = (k > i ? f[i][k - 1] : 0) + f[k + 1][j - 1] + 2 * gamma * probs[k][j];
                        best = Math.max(best, v);
                    }
                }
                f[i][j] = best;
            }
        }
        int[] pairs = new int[n];
        Arrays.fill(pairs, -1);
        traceMea(probs, q, f, gamma, 0, n - 1, pairs);
        return pairs;
    }

    private static void traceMea(double[][] probs, double[] q, double[][] f, double gamma, int i, int j, int[] pairs) {
        while (j >= i) {
            double unp = (j > i ? f[i][j - 1] : 0) + q[j];
            if (Math.abs(f[i][j] - unp) < 1e-12) {
                j--;
                continue;
            }
            int found = -1;
            for (int k = i; k < j - MIN_HAIRPIN; k++) {
                if (probs[k][j] > 0) {
                    double v = (k > i ? f[i][k - 1] : 0) + f[k + 1][j - 1] + 2 * gamma * probs[k][j];
                    if (Math.abs(v - f[i][j]) < 1e-12) {
                        found = k;
                        break;
                    }
                }
            }
            if (found < 0) {
                j--;
                continue;
            }
            pairs[found] = j;
            pairs[j] = found;
            traceMea(probs, q, f, gamma, found + 1, j - 1, pairs);
            j = found - 1;
        }
    }

    /**
     * Trains the weights as a CRF: maximises {@code sum log P(true structure | sequence)} with an
     * L2 penalty by Adam; the gradient of each example is the true structure's feature counts
     * minus the expected counts under the model (from the inside-outside posteriors). Examples
     * whose structure the grammar cannot derive are skipped.
     *
     * @param data training sequences with structures
     * @param init the initial weights (copied)
     * @param epochs passes over the data
     * @param learningRate the Adam step size
     * @param l2 the L2 penalty
     * @return the learned weights
     */
    public static double[] trainCrf(List<BioSequences.RnaStructure> data, double[] init, int epochs,
            double learningRate, double l2) {
        double[] w = init.clone();
        double[] m = new double[FEATURES];
        double[] v = new double[FEATURES];
        List<Grammar> grammars = new ArrayList<>();
        List<double[]> truthFeats = new ArrayList<>();
        for (BioSequences.RnaStructure s : data) {
            Grammar g = new Grammar(s.sequence());
            int n = s.sequence().length;
            int[] pref = new int[n + 1];
            for (int i = 0; i < n; i++) {
                pref[i + 1] = pref[i] + (s.pairs()[i] >= 0 ? 1 : 0);
            }
            boolean[] allowed = new boolean[g.edges];
            for (int e = 0; e < g.edges; e++) {
                allowed[e] = g.consistent(e, s.pairs(), pref);
            }
            int[] arg = new int[g.cells];
            double[] in = g.inside(w, allowed, true, arg);
            if (in[g.eCell[n]] == Double.NEGATIVE_INFINITY) {
                continue;
            }
            int[] pairs = new int[n];
            Arrays.fill(pairs, -1);
            double[] feats = new double[FEATURES];
            g.collect(g.eCell[n], arg, pairs, feats);
            if (!Arrays.equals(pairs, s.pairs())) {
                continue;
            }
            grammars.add(g);
            truthFeats.add(feats);
        }
        int step = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int idx = 0; idx < grammars.size(); idx++) {
                Grammar g = grammars.get(idx);
                double[] in = g.inside(w, null, false, null);
                double[] post = g.posteriors(w, null, in);
                double[] grad = new double[FEATURES];
                for (int e = 0; e < g.edges; e++) {
                    if (post[e] == 0) {
                        continue;
                    }
                    for (int f = g.featStart[e]; f < g.featStart[e + 1]; f++) {
                        grad[g.featIdx[f]] += post[e] * g.featCnt[f];
                    }
                }
                step++;
                for (int f = 0; f < FEATURES; f++) {
                    double gf = (grad[f] - truthFeats.get(idx)[f]) / g.n + l2 * (w[f] - init[f]);
                    m[f] = 0.9 * m[f] + 0.1 * gf;
                    v[f] = 0.999 * v[f] + 0.001 * gf * gf;
                    w[f] -= learningRate * (m[f] / (1 - Math.pow(0.9, step)))
                            / (Math.sqrt(v[f] / (1 - Math.pow(0.999, step))) + 1e-8);
                }
            }
        }
        return w;
    }

    /**
     * Base-pair F1 against the true structure.
     *
     * @param predicted predicted partners
     * @param truth true partners
     * @return the F1 of the pair sets
     */
    public static double pairF1(int[] predicted, int[] truth) {
        int tp = 0;
        int np = 0;
        int nt = 0;
        for (int i = 0; i < truth.length; i++) {
            if (predicted[i] > i) {
                np++;
                tp += truth[i] == predicted[i] ? 1 : 0;
            }
            if (truth[i] > i) {
                nt++;
            }
        }
        return tp == 0 ? 0 : 2.0 * tp / (np + nt);
    }

    /**
     * Whether a pairing is a valid secondary structure: symmetric, canonical, nested (no
     * pseudoknots) and with hairpins of at least {@link #MIN_HAIRPIN}.
     *
     * @param seq the sequence
     * @param pairs the partner array
     * @return validity
     */
    public static boolean isValid(int[] seq, int[] pairs) {
        for (int i = 0; i < pairs.length; i++) {
            int j = pairs[i];
            if (j < 0) {
                continue;
            }
            if (pairs[j] != i || Math.abs(j - i) <= MIN_HAIRPIN || pairType(seq[Math.min(i, j)], seq[Math.max(i, j)]) < 0) {
                return false;
            }
            if (j > i) {
                for (int k = i + 1; k < j; k++) {
                    if (pairs[k] >= 0 && (pairs[k] < i || pairs[k] > j)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * The unconstrained baseline: windowed one-hot residues (width 7) through a tanh layer give
     * per-position embeddings, pair logits are their scaled dot products plus a bias (a symmetric
     * output), trained with the logistic loss on the pairing matrix; decoding greedily accepts
     * the most probable pairs above one half while each base is still free, so the result may
     * be non-canonical or pseudoknotted.
     */
    public static final class PairCnn {
        private static final int WINDOW = 7;
        private static final float POS_WEIGHT = 10;
        private final Var w1;
        private final Var b1;
        private final Var bias;
        private final int dim;

        /**
         * Creates an untrained model.
         *
         * @param dim the embedding dimension
         * @param seed the seed
         */
        public PairCnn(int dim, long seed) {
            this.dim = dim;
            SplittableRandom rnd = new SplittableRandom(seed);
            float[] w = new float[WINDOW * 4 * dim];
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) ((rnd.nextDouble() - 0.5) * 0.6);
            }
            w1 = Var.param(Tensor.of(Shape.of(WINDOW * 4, dim), w));
            b1 = Var.param(Tensor.zeros(Shape.of(dim)));
            bias = Var.param(Tensor.of(Shape.of(1), -3f));
        }

        private Var logits(int[] seq) {
            int n = seq.length;
            float[] x = new float[n * WINDOW * 4];
            for (int i = 0; i < n; i++) {
                for (int o = 0; o < WINDOW; o++) {
                    int p = i + o - WINDOW / 2;
                    if (p >= 0 && p < n) {
                        x[(i * WINDOW + o) * 4 + seq[p]] = 1;
                    }
                }
            }
            Var h = Var.of(Tensor.of(Shape.of(n, WINDOW * 4), x)).matmul(w1).add(b1).tanh();
            return h.matmul(h.transpose()).mul((float) (2.0 / Math.sqrt(dim))).add(bias);
        }

        /**
         * Trains by Adam, one sequence per step.
         *
         * @param data training structures
         * @param steps gradient steps
         * @param learningRate the Adam step size
         * @param seed the sampling seed
         */
        public void fit(List<BioSequences.RnaStructure> data, int steps, float learningRate, long seed) {
            Optimizer opt = Optimizer.adam(learningRate);
            SplittableRandom rnd = new SplittableRandom(seed);
            for (int s = 0; s < steps; s++) {
                BioSequences.RnaStructure ex = data.get(rnd.nextInt(data.size()));
                int n = ex.sequence().length;
                float[] y = new float[n * n];
                for (int i = 0; i < n; i++) {
                    if (ex.pairs()[i] >= 0) {
                        y[i * n + ex.pairs()[i]] = 1;
                    }
                }
                Var z = logits(ex.sequence());
                // positives are rare: weight them by POS_WEIGHT in the logistic loss
                float[] wy = new float[n * n];
                float[] wAll = new float[n * n];
                for (int i = 0; i < n * n; i++) {
                    wy[i] = POS_WEIGHT * y[i];
                    wAll[i] = 1 - y[i] + POS_WEIGHT * y[i];
                }
                Var loss = z.softplus().mul(Tensor.of(Shape.of(n, n), wAll)).sub(z.mul(Tensor.of(Shape.of(n, n), wy)))
                        .mean();
                loss.backward();
                opt.step(List.of(w1, b1, bias));
            }
        }

        /**
         * Predicts a pairing by greedy decoding.
         *
         * @param seq the sequence
         * @return the partner array (possibly invalid)
         */
        public int[] predict(int[] seq) {
            int n = seq.length;
            float[] z = Var.noGrad(() -> logits(seq)).value().toArray();
            List<int[]> cand = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (z[i * n + j] > 0) {
                        cand.add(new int[] {i, j});
                    }
                }
            }
            cand.sort((a, b) -> Float.compare(z[b[0] * n + b[1]], z[a[0] * n + a[1]]));
            int[] pairs = new int[n];
            Arrays.fill(pairs, -1);
            for (int[] c : cand) {
                if (pairs[c[0]] < 0 && pairs[c[1]] < 0) {
                    pairs[c[0]] = c[1];
                    pairs[c[1]] = c[0];
                }
            }
            return pairs;
        }
    }
}
