package io.github.zzuegg.brainiac.experimental.parse;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Unsupervised constituency parsing with a probabilistic context-free grammar in Chomsky normal
 * form, trained on the marginal likelihood of raw sentences (the inside algorithm written in
 * {@link Var} ops, in log space, so gradients are the expected rule counts of EM).
 *
 * <ul>
 *   <li>{@link Parametrisation#FLAT}: free logits per rule, the classic PCFG (gradient EM);
 *   <li>{@link Parametrisation#NEURAL}: rule, root and emission distributions computed from
 *       symbol embeddings by small residual MLPs, sharing statistics across symbols;
 *   <li>{@link Parametrisation#COMPOUND}: the neural grammar conditioned on a per-sentence
 *       latent {@code z} (amortised Gaussian posterior from mean word embeddings, KL to the
 *       standard normal; {@code z = mu} at test time).
 * </ul>
 *
 * <p>Symbols are {@code nt} nonterminals (for spans of two or more words) and {@code pt}
 * preterminals (single words). {@link #parse} decodes by Viterbi CKY or by minimum Bayes risk
 * over span marginals, which are the gradients of the log-partition with respect to a zero bias
 * on every span.
 *
 * <p>Paper: Kim, Dyer, Rush, "Compound Probabilistic Context-Free Grammars for Grammar
 * Induction", ACL 2019, <a href="https://arxiv.org/abs/1906.10225">arXiv:1906.10225</a>.
 */
public final class NeuralPcfg {

    /** How rule probabilities are produced. */
    public enum Parametrisation {
        /** free logits */
        FLAT,
        /** MLPs over symbol embeddings */
        NEURAL,
        /** the neural grammar with a per-sentence latent */
        COMPOUND
    }

    /** Decoders. */
    public enum Decoding {
        /** the single most likely tree */
        VITERBI,
        /** the tree with the largest summed span marginals */
        MBR
    }

    private static final float NEG = -1e4f;

    private final Parametrisation kind;
    private final int nt;
    private final int pt;
    private final int vocab;
    private final int dim;
    private final int zDim;
    private final List<Var> params = new ArrayList<>();
    // flat
    private Var flatRoot;
    private Var flatRule;
    private Var flatEmit;
    // neural
    private Var symbols;
    private Var rootW;
    private Var rootOut;
    private Var ruleW;
    private Var ruleOut;
    private Var emitW;
    private Var emitOut;
    private Var wordEmbed;
    private Var encMu;
    private Var encLogVar;

    /**
     * Creates a grammar.
     *
     * @param kind the parametrisation
     * @param nonterminals nonterminal symbols
     * @param preterminals preterminal symbols
     * @param vocab the vocabulary
     * @param dim the symbol embedding width (neural kinds)
     * @param zDim the latent width ({@link Parametrisation#COMPOUND})
     * @param seed the seed
     */
    public NeuralPcfg(Parametrisation kind, int nonterminals, int preterminals, int vocab, int dim, int zDim,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.kind = kind;
        this.nt = nonterminals;
        this.pt = preterminals;
        this.vocab = vocab;
        this.dim = dim;
        this.zDim = kind == Parametrisation.COMPOUND ? zDim : 0;
        int s = nt + pt;
        if (kind == Parametrisation.FLAT) {
            flatRoot = param(r, 0.5f, nt);
            flatRule = param(r, 0.5f, nt, s * s);
            flatEmit = param(r, 0.5f, pt, vocab);
        } else {
            int in = dim + this.zDim;
            symbols = param(r, 0.5f, nt + pt + 1, dim);
            rootW = param(r, (float) Math.sqrt(1.0 / in), in, dim);
            rootOut = param(r, (float) Math.sqrt(1.0 / dim), dim, nt);
            ruleW = param(r, (float) Math.sqrt(1.0 / in), in, dim);
            ruleOut = param(r, (float) Math.sqrt(1.0 / dim), dim, s * s);
            emitW = param(r, (float) Math.sqrt(1.0 / in), in, dim);
            emitOut = param(r, (float) Math.sqrt(1.0 / dim), dim, vocab);
            if (kind == Parametrisation.COMPOUND) {
                wordEmbed = param(r, 0.3f, vocab, dim);
                encMu = param(r, (float) Math.sqrt(1.0 / dim), dim, this.zDim);
                encLogVar = param(r, (float) Math.sqrt(1.0 / dim), dim, this.zDim);
            }
        }
    }

    private Var param(SplittableRandom r, float std, int... dims) {
        Var v = Var.param(Tensor.normal(Shape.of(dims), 0, std, r));
        params.add(v);
        return v;
    }

    /** @return the trainable values */
    public List<Var> parameters() {
        return List.copyOf(params);
    }

    /** Log-probabilities for a batch: root [b, nt], rule [b, nt, s*s], emit [b, pt, vocab]. */
    private record Grammar(Var root, Var rule, Var emit, Var kl) {
    }

    private static Var ids(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Var.of(Tensor.of(Shape.of(v.length), f));
    }

    private static Tensor range(int from, int to) {
        float[] f = new float[to - from];
        for (int i = 0; i < f.length; i++) {
            f[i] = from + i;
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /** {@code [rows, dim]} symbol embeddings with {@code z [b, zDim]} appended per batch row. */
    private Var condition(Var e, Var z, int b) {
        int rows = e.shape().dim(0);
        Var x = e.reshape(1, rows, dim);
        if (z == null) {
            return x;
        }
        Var tiled = x.add(Var.of(Tensor.zeros(Shape.of(b, rows, dim))));
        Var zt = z.reshape(b, 1, zDim).add(Var.of(Tensor.zeros(Shape.of(b, rows, zDim))));
        return Var.concat(2, tiled, zt);
    }

    private Var mlp(Var x, Var w, Var out, Var e, int b) {
        Var h = x.matmul(w).relu().add(e.reshape(1, e.shape().dim(0), dim));
        return h.matmul(out);
    }

    private Grammar grammar(int[][] words, boolean sample, SplittableRandom r) {
        int b = words.length;
        int s = nt + pt;
        if (kind == Parametrisation.FLAT) {
            Var root = flatRoot.logSoftmax(0).reshape(1, nt);
            Var rule = flatRule.logSoftmax(1).reshape(1, nt, s * s);
            Var emit = flatEmit.logSoftmax(1).reshape(1, pt, vocab);
            return new Grammar(root, rule, emit, null);
        }
        Var z = null;
        Var kl = null;
        if (kind == Parametrisation.COMPOUND) {
            int n = words[0].length;
            int[] flat = new int[b * n];
            for (int i = 0; i < b; i++) {
                System.arraycopy(words[i], 0, flat, i * n, n);
            }
            Var mean = wordEmbed.gather(0, ids(flat).value()).reshape(b, n, dim).mean(1, false);
            Var mu = mean.matmul(encMu);
            Var logVar = mean.matmul(encLogVar).tanh().mul(3f);
            z = mu;
            if (sample) {
                Tensor eps = Tensor.normal(Shape.of(b, zDim), 0, 1, r);
                z = mu.add(logVar.mul(0.5f).exp().mul(eps));
            }
            kl = mu.square().add(logVar.exp()).sub(logVar).sub(1f).mul(0.5f).sum(1, false);
        }
        Var eRoot = symbols.gather(0, range(nt + pt, nt + pt + 1));
        Var eNt = symbols.gather(0, range(0, nt));
        Var ePt = symbols.gather(0, range(nt, nt + pt));
        Var root = mlp(condition(eRoot, z, b), rootW, rootOut, eRoot, b).logSoftmax(2);
        Var rule = mlp(condition(eNt, z, b), ruleW, ruleOut, eNt, b).logSoftmax(2);
        Var emit = mlp(condition(ePt, z, b), emitW, emitOut, ePt, b).logSoftmax(2);
        return new Grammar(root.reshape(root.shape().dim(0), nt), rule, emit, kl);
    }

    /** Inside pass; returns log Z [b] and, when asked, the span bias variables. */
    private Var inside(Grammar g, int[][] words, Map<Long, Var> biases) {
        int b = words.length;
        int n = words[0].length;
        int s = nt + pt;
        Var[][] chart = new Var[n][n + 1];
        Var padNt = Var.of(Tensor.full(Shape.of(b, nt), NEG, g.emit().value().device()));
        Var padPt = Var.of(Tensor.full(Shape.of(b, pt), NEG, g.emit().value().device()));
        for (int i = 0; i < n; i++) {
            float[] oneHot = new float[b * vocab];
            for (int k = 0; k < b; k++) {
                oneHot[k * vocab + words[k][i]] = 1f;
            }
            // emit [gb, pt, vocab] x onehot [b, vocab, 1] -> [b, pt]
            Var e = g.emit().mul(Var.of(Tensor.of(Shape.of(b, 1, vocab), oneHot))).sum(2, false);
            chart[i][i + 1] = Var.concat(1, padNt, e.reshape(b, pt));
        }
        for (int w = 2; w <= n; w++) {
            for (int i = 0; i + w <= n; i++) {
                int j = i + w;
                Var[] splits = new Var[w - 1];
                for (int k = i + 1; k < j; k++) {
                    Var l = chart[i][k].reshape(b, s, 1);
                    Var rr = chart[k][j].reshape(b, 1, s);
                    splits[k - i - 1] = l.add(rr).reshape(b, 1, s * s);
                }
                Var children = (splits.length == 1 ? splits[0] : Var.concat(1, splits).logsumexp(1, true))
                        .reshape(b, 1, s * s);
                Var in = g.rule().add(children).logsumexp(2, false);       // [b, nt]
                if (biases != null) {
                    Var bias = Var.param(Tensor.zeros(Shape.of(b, 1)));
                    biases.put(key(i, j), bias);
                    in = in.add(bias);
                }
                chart[i][j] = Var.concat(1, in, padPt);
            }
        }
        Var top = chart[0][n].slice(1, 0, nt);
        return top.add(g.root()).logsumexp(1, false);
    }

    private static long key(int i, int j) {
        return ((long) i << 32) | j;
    }

    /**
     * Trains on raw sentences (batches share a length).
     *
     * @param sentences word ids
     * @param steps steps
     * @param batch batch size
     * @param lr the learning rate
     * @param seed the seed
     */
    public void train(List<int[]> sentences, int steps, int batch, float lr, long seed) {
        Map<Integer, List<int[]>> buckets = new HashMap<>();
        for (int[] x : sentences) {
            if (x.length >= 2) {
                buckets.computeIfAbsent(x.length, k -> new ArrayList<>()).add(x);
            }
        }
        List<Integer> lengths = new ArrayList<>(buckets.keySet());
        List<Integer> weights = new ArrayList<>();
        for (int l : lengths) {
            weights.add(buckets.get(l).size());
        }
        int total = weights.stream().mapToInt(Integer::intValue).sum();
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        for (int step = 0; step < steps; step++) {
            int pick = r.nextInt(total);
            int li = 0;
            while (pick >= weights.get(li)) {
                pick -= weights.get(li++);
            }
            List<int[]> bucket = buckets.get(lengths.get(li));
            int b = Math.min(batch, bucket.size());
            int[][] words = new int[b][];
            for (int i = 0; i < b; i++) {
                words[i] = bucket.get(r.nextInt(bucket.size()));
            }
            Grammar g = grammar(words, true, r);
            Var nll = inside(g, words, null).mul(-1f);
            if (g.kl() != null) {
                nll = nll.add(g.kl());
            }
            params.forEach(Var::zeroGrad);
            nll.mean().div(words[0].length).backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
    }

    /**
     * The log-likelihood of a sentence ({@code z = mu} for the compound grammar, so a lower
     * bound up to the KL term).
     *
     * @param words word ids
     * @return {@code log p(words)}
     */
    public double logLikelihood(int[] words) {
        int[][] w = {words};
        return Var.noGrad(() -> inside(grammar(w, false, null), w, null)).value().toArray()[0];
    }

    /**
     * Parses a sentence.
     *
     * @param words word ids
     * @param how the decoder
     * @return unlabelled spans {@code {start, end}} of width two or more, the whole sentence
     *     included
     */
    public List<int[]> parse(int[] words, Decoding how) {
        int n = words.length;
        List<int[]> out = new ArrayList<>();
        if (n < 2) {
            return out;
        }
        int[][] w = {words};
        double[][] score = new double[n][n + 1];
        if (how == Decoding.MBR) {
            Map<Long, Var> biases = new HashMap<>();
            Grammar g = grammar(w, false, null);
            Var z = inside(g, w, biases);
            z.sum().backward();
            for (Map.Entry<Long, Var> e : biases.entrySet()) {
                int i = (int) (e.getKey() >> 32);
                int j = (int) (e.getKey() & 0xffffffffL);
                score[i][j] = e.getValue().grad().toArray()[0];
            }
            params.forEach(Var::zeroGrad);
            spans(bestTree(score, n), 0, n, out);
            return out;
        }
        Grammar g = Var.noGrad(() -> grammar(w, false, null));
        int s = nt + pt;
        float[] root = g.root().value().toArray();
        float[] rule = g.rule().value().toArray();
        float[] emit = g.emit().value().toArray();
        double[][][] best = new double[n][n + 1][];
        int[][][] split = new int[n][n + 1][];
        int[][][] left = new int[n][n + 1][];
        int[][][] right = new int[n][n + 1][];
        for (int i = 0; i < n; i++) {
            best[i][i + 1] = new double[s];
            java.util.Arrays.fill(best[i][i + 1], NEG);
            for (int p = 0; p < pt; p++) {
                best[i][i + 1][nt + p] = emit[p * vocab + words[i]];
            }
        }
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len <= n; i++) {
                int j = i + len;
                double[] v = new double[s];
                java.util.Arrays.fill(v, NEG);
                int[] sp = new int[s];
                int[] lb = new int[s];
                int[] rc = new int[s];
                for (int k = i + 1; k < j; k++) {
                    double[] l = best[i][k];
                    double[] rr = best[k][j];
                    for (int bb = 0; bb < s; bb++) {
                        if (l[bb] <= NEG / 2) {
                            continue;
                        }
                        for (int c = 0; c < s; c++) {
                            if (rr[c] <= NEG / 2) {
                                continue;
                            }
                            double lr = l[bb] + rr[c];
                            for (int a = 0; a < nt; a++) {
                                double cand = lr + rule[a * s * s + bb * s + c];
                                if (cand > v[a]) {
                                    v[a] = cand;
                                    sp[a] = k;
                                    lb[a] = bb;
                                    rc[a] = c;
                                }
                            }
                        }
                    }
                }
                best[i][j] = v;
                split[i][j] = sp;
                left[i][j] = lb;
                right[i][j] = rc;
            }
        }
        int a = 0;
        for (int k = 1; k < nt; k++) {
            if (best[0][n][k] + root[k] > best[0][n][a] + root[a]) {
                a = k;
            }
        }
        backtrack(0, n, a, split, left, right, out);
        return out;
    }

    private static void backtrack(int i, int j, int a, int[][][] split, int[][][] left, int[][][] right,
            List<int[]> out) {
        if (j - i < 2) {
            return;
        }
        out.add(new int[] {i, j});
        int k = split[i][j][a];
        backtrack(i, k, left[i][j][a], split, left, right, out);
        backtrack(k, j, right[i][j][a], split, left, right, out);
    }

    /** CKY over span scores: the binary tree maximising summed marginals; returns split points. */
    private static int[][] bestTree(double[][] score, int n) {
        double[][] v = new double[n][n + 1];
        int[][] sp = new int[n][n + 1];
        for (int len = 2; len <= n; len++) {
            for (int i = 0; i + len <= n; i++) {
                int j = i + len;
                double best = Double.NEGATIVE_INFINITY;
                for (int k = i + 1; k < j; k++) {
                    double c = v[i][k] + v[k][j];
                    if (c > best) {
                        best = c;
                        sp[i][j] = k;
                    }
                }
                v[i][j] = best + score[i][j];
            }
        }
        return sp;
    }

    private static void spans(int[][] sp, int i, int j, List<int[]> out) {
        if (j - i < 2) {
            return;
        }
        out.add(new int[] {i, j});
        spans(sp, i, sp[i][j], out);
        spans(sp, sp[i][j], j, out);
    }
}
