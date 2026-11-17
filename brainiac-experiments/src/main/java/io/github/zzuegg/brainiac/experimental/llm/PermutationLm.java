package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * An XLNet-style permutation language model: autoregressive training over a random
 * factorisation order {@code z} with two-stream self-attention.
 *
 * <ul>
 *   <li>The content stream {@code h_i} starts from the token embedding plus position and
 *       attends to every position not later than {@code i} in the order (itself included).</li>
 *   <li>The query stream {@code g_i} starts from a learned vector plus position only and attends
 *       to the content of the positions strictly before {@code i}; it predicts {@code x_i}
 *       without seeing it, yet knows where it is.</li>
 *   <li>A learned null key/value is visible to both streams, so the first position in the order
 *       has something to attend to.</li>
 * </ul>
 *
 * Both streams share the weights of each post-norm transformer layer. Only the last
 * {@code predict} positions of each order are trained (XLNet's partial prediction). For
 * downstream use, {@link #features} runs the content stream with full bidirectional attention.
 *
 * <p>Paper: Yang et al., "XLNet: Generalized Autoregressive Pretraining for Language
 * Understanding" (NeurIPS 2019), <a href="https://arxiv.org/abs/1906.08237">arXiv:1906.08237</a>.
 */
public final class PermutationLm {

    private static final float NEG = -1e9f;

    private final int vocab;
    private final int length;
    private final int width;
    private final int heads;
    private final Var embedding;
    private final Var position;
    private final Var queryStart;
    private final List<Var[]> layers = new ArrayList<>();
    private final List<LayerNorm[]> norms = new ArrayList<>();
    private final Var output;
    private final Var outputBias;

    private PermutationLm(int vocab, int length, int width, int heads, int depth, long seed) {
        if (width % heads != 0) {
            throw new IllegalArgumentException("heads must divide width");
        }
        SplittableRandom r = new SplittableRandom(seed);
        this.vocab = vocab;
        this.length = length;
        this.width = width;
        this.heads = heads;
        this.embedding = Var.param(Tensor.normal(Shape.of(vocab, width), 0, 0.3f, r));
        this.position = Var.param(Tensor.normal(Shape.of(1, length, width), 0, 0.3f, r));
        this.queryStart = Var.param(Tensor.normal(Shape.of(1, 1, width), 0, 0.3f, r));
        for (int i = 0; i < depth; i++) {
            layers.add(new Var[] {matrix(r, width, width), matrix(r, width, width), matrix(r, width, width),
                matrix(r, width, width), Var.param(Tensor.normal(Shape.of(1, 1, width), 0, 0.3f, r)),
                Var.param(Tensor.normal(Shape.of(1, 1, width), 0, 0.3f, r)), matrix(r, width, 2 * width),
                Var.param(Tensor.zeros(Shape.of(2 * width))), matrix(r, 2 * width, width),
                Var.param(Tensor.zeros(Shape.of(width)))});
            norms.add(new LayerNorm[] {LayerNorm.of(width), LayerNorm.of(width)});
        }
        this.output = matrix(r, width, vocab);
        this.outputBias = Var.param(Tensor.zeros(Shape.of(vocab)));
    }

    /**
     * Creates the model.
     *
     * @param vocab the vocabulary
     * @param length the sequence length
     * @param width the model width
     * @param heads the attention heads
     * @param depth the layers
     * @param seed the initialization seed
     * @return the model
     */
    public static PermutationLm of(int vocab, int length, int width, int heads, int depth, long seed) {
        return new PermutationLm(vocab, length, width, heads, depth, seed);
    }

    private static Var matrix(SplittableRandom r, int in, int out) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(2.0 / (in + out)), r));
    }

    private Tensor ids(int[][] tokens) {
        float[] f = new float[tokens.length * length];
        for (int i = 0; i < tokens.length; i++) {
            for (int t = 0; t < length; t++) {
                f[i * length + t] = tokens[i][t];
            }
        }
        return Tensor.of(Shape.of(tokens.length, length), f);
    }

    /**
     * Additive attention masks {@code [b, heads, L, L + 1]} (key 0 is the null slot) for orders
     * given as ranks: {@code rank[b][i]} is position {@code i}'s place in the order.
     */
    private Tensor mask(int[][] rank, boolean strict) {
        int b = rank.length;
        int keys = length + 1;
        float[] m = new float[b * heads * length * keys];
        for (int n = 0; n < b; n++) {
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < length; j++) {
                    boolean visible = strict ? rank[n][j] < rank[n][i] : rank[n][j] <= rank[n][i];
                    if (!visible) {
                        for (int h = 0; h < heads; h++) {
                            m[((n * heads + h) * length + i) * keys + j + 1] = NEG;
                        }
                    }
                }
            }
        }
        return Tensor.of(Shape.of(b, heads, length, keys), m);
    }

    private Var attend(Var queries, Var content, Tensor mask, Var[] w) {
        int b = queries.shape().dim(0);
        int dh = width / heads;
        Var q = queries.matmul(w[0]).reshape(b, length, heads, dh).permute(0, 2, 1, 3);
        Var k = Var.concat(1, w[4].expand(Shape.of(b, 1, width)), content).matmul(w[1])
                .reshape(b, length + 1, heads, dh).permute(0, 2, 1, 3);
        Var v = Var.concat(1, w[5].expand(Shape.of(b, 1, width)), content).matmul(w[2])
                .reshape(b, length + 1, heads, dh).permute(0, 2, 1, 3);
        Var scores = q.matmul(k.permute(0, 1, 3, 2)).mul((float) (1 / Math.sqrt(dh))).add(mask);
        Var mixed = Activation.SOFTMAX.forward(scores).matmul(v).permute(0, 2, 1, 3).reshape(b, length, width);
        return mixed.matmul(w[3]);
    }

    private Var block(Var x, Var attention, int layer) {
        Var[] w = layers.get(layer);
        LayerNorm[] n = norms.get(layer);
        Var h = n[0].forward(x.add(attention));
        Var f = Activation.GELU.forward(h.matmul(w[6]).add(w[7])).matmul(w[8]).add(w[9]);
        return n[1].forward(h.add(f));
    }

    /**
     * Runs both streams.
     *
     * @param tokens {@code [b][L]} ids (tokens later in the order may be anything for the
     *     query stream, which never sees them)
     * @param rank each position's place in the factorisation order
     * @param training whether training
     * @return {@code {content [b, L, width], query [b, L, width]}}
     */
    private Var[] streams(int[][] tokens, int[][] rank, boolean training) {
        int b = tokens.length;
        Var h = embedding.gather(0, ids(tokens)).add(position.expand(Shape.of(b, length, width)));
        Var g = queryStart.expand(Shape.of(b, length, width)).add(position.expand(Shape.of(b, length, width)));
        Tensor contentMask = mask(rank, false);
        Tensor queryMask = mask(rank, true);
        for (int l = 0; l < layers.size(); l++) {
            Var[] w = layers.get(l);
            Var nextH = block(h, attend(h, h, contentMask, w), l);
            g = block(g, attend(g, h, queryMask, w), l);
            h = nextH;
        }
        return new Var[] {h, g};
    }

    /**
     * The permutation-LM loss on one batch: a random order per sequence, cross-entropy of the
     * query stream at the last {@code predict} positions of each order.
     *
     * @param tokens {@code [b][L]} ids
     * @param predict positions per sequence to predict
     * @param random the source of orders
     * @return the mean loss
     */
    public Var loss(int[][] tokens, int predict, RandomGenerator random) {
        int b = tokens.length;
        int[][] rank = new int[b][];
        float[] target = new float[b * length];
        for (int n = 0; n < b; n++) {
            rank[n] = randomRanks(random);
            for (int i = 0; i < length; i++) {
                target[n * length + i] = rank[n][i] >= length - predict ? tokens[n][i] : -1;
            }
        }
        Var g = streams(tokens, rank, true)[1];
        Var logits = g.reshape(b * length, width).matmul(output).add(outputBias);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(b * length), target));
    }

    private int[] randomRanks(RandomGenerator random) {
        int[] order = new int[length];
        for (int i = 0; i < length; i++) {
            order[i] = i;
        }
        for (int i = length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        int[] rank = new int[length];
        for (int i = 0; i < length; i++) {
            rank[order[i]] = i;
        }
        return rank;
    }

    /**
     * The content stream with full bidirectional attention, as downstream features.
     *
     * @param tokens {@code [b][L]} ids
     * @return {@code [b, L, width]}
     */
    public Var features(int[][] tokens) {
        int[][] rank = new int[tokens.length][length];   // every rank equal: all positions visible
        return streams(tokens, rank, false)[0];
    }

    /**
     * Fills missing positions, left to right, each conditioned on the visible tokens and the
     * ones already filled: the order puts visible positions first and missing ones last.
     *
     * @param tokens {@code [b][L]} ids; entries at missing positions are ignored
     * @param missing which positions to fill
     * @return the completed sequences
     */
    public int[][] infill(int[][] tokens, boolean[][] missing) {
        int b = tokens.length;
        int[][] out = new int[b][];
        int[][] rank = new int[b][length];
        int maxMissing = 0;
        for (int n = 0; n < b; n++) {
            out[n] = tokens[n].clone();
            int visible = 0;
            for (int i = 0; i < length; i++) {
                visible += missing[n][i] ? 0 : 1;
            }
            int v = 0;
            int m = visible;
            for (int i = 0; i < length; i++) {
                if (missing[n][i]) {
                    out[n][i] = 0;
                    rank[n][i] = m++;
                } else {
                    rank[n][i] = v++;
                }
            }
            maxMissing = Math.max(maxMissing, length - visible);
        }
        for (int k = 0; k < maxMissing; k++) {
            float[] logits = streams(out, rank, false)[1].reshape(b * length, width).matmul(output).add(outputBias)
                    .value().toArray();
            for (int n = 0; n < b; n++) {
                int visible = 0;
                for (int i = 0; i < length; i++) {
                    visible += missing[n][i] ? 0 : 1;
                }
                for (int i = 0; i < length; i++) {
                    if (missing[n][i] && rank[n][i] == visible + k) {
                        int best = 0;
                        for (int c = 1; c < vocab; c++) {
                            if (logits[(n * length + i) * vocab + c] > logits[(n * length + i) * vocab + best]) {
                                best = c;
                            }
                        }
                        out[n][i] = best;
                    }
                }
            }
        }
        return out;
    }

    /** @return every trainable value */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding, position, queryStart, output, outputBias));
        for (Var[] w : layers) {
            p.addAll(List.of(w));
        }
        for (LayerNorm[] n : norms) {
            p.addAll(n[0].parameters());
            p.addAll(n[1].parameters());
        }
        return p;
    }

    /** @return the number of trainable scalars */
    public long parameterCount() {
        long c = 0;
        for (Var p : parameters()) {
            c += p.shape().size();
        }
        return c;
    }

    @Override
    public String toString() {
        return "PermutationLm(" + vocab + " tokens, " + length + " positions, " + layers.size() + " x " + width + ")";
    }
}
