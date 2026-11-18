package io.github.zzuegg.brainiac.experimental.document;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LinearChainCrf;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A layout-aware token encoder in the style of LayoutLM (Xu et al., KDD 2020). Forms carry
 * meaning in their layout: a value sits to the right of or below its key. Each token's
 * embedding is the sum of its word embedding, its 1-D position (reading order) embedding
 * and, when 2-D positions are on, embeddings of its bounding box: {@code x0} and {@code x1}
 * from one learned table, {@code y0} and {@code y1} from another, plus width and height
 * tables. A stack of core {@link TransformerBlock}s (bidirectional) encodes the sequence;
 * key-value extraction is then token tagging with a {@link LinearChainCrf} on top.
 *
 * <p>{@link #maskedLoss} is the masked visual-language model pretraining of the paper:
 * replace 15% of the tokens by a mask id (keeping their boxes) and predict them, so the
 * model learns to use layout to fill in text.
 *
 * <p>Box coordinates are integers in {@code [0, grid)}; token id {@code vocab - 1} is
 * reserved as the mask.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1912.13318">arXiv:1912.13318</a>.
 */
public final class LayoutEncoder {

    private final int vocab;
    private final int dim;
    private final int grid;
    private final boolean twoD;
    private final Var words;
    private final Var order;
    private final Var xs;
    private final Var ys;
    private final Var ws;
    private final Var hs;
    private final List<Layer> blocks = new ArrayList<>();
    private final Var tagW;
    private final Var tagB;
    private final Var mlmW;
    private final Var mlmB;
    private final LinearChainCrf crf;

    /**
     * Creates the encoder.
     *
     * @param vocab word ids, the last being the mask
     * @param tags output tags
     * @param dim the model width
     * @param layers transformer blocks
     * @param maxTokens the longest sequence
     * @param grid the box coordinate range
     * @param twoD whether to add the 2-D box embeddings
     * @param seed the seed
     */
    public LayoutEncoder(int vocab, int tags, int dim, int layers, int maxTokens, int grid, boolean twoD, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.vocab = vocab;
        this.dim = dim;
        this.grid = grid;
        this.twoD = twoD;
        this.words = table(r, vocab);
        this.order = table(r, maxTokens);
        this.xs = table(r, grid);
        this.ys = table(r, grid);
        this.ws = table(r, grid);
        this.hs = table(r, grid);
        for (int i = 0; i < layers; i++) {
            blocks.add(TransformerBlock.of(4).ffnSize(2 * dim).create(Shape.of(8, dim), r));
        }
        this.tagW = Var.param(Tensor.normal(Shape.of(dim, tags), 0, (float) Math.sqrt(1.0 / dim), r));
        this.tagB = Var.param(Tensor.zeros(Shape.of(tags)));
        this.mlmW = Var.param(Tensor.normal(Shape.of(dim, vocab), 0, (float) Math.sqrt(1.0 / dim), r));
        this.mlmB = Var.param(Tensor.zeros(Shape.of(vocab)));
        this.crf = new LinearChainCrf(tags, r);
    }

    private Var table(SplittableRandom r, int n) {
        return Var.param(Tensor.normal(Shape.of(n, dim), 0, 0.3f, r));
    }

    private static Tensor ids(int[][] v, int clampMax) {
        int b = v.length;
        int n = v[0].length;
        float[] f = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < n; t++) {
                f[i * n + t] = Math.max(0, Math.min(clampMax, v[i][t]));
            }
        }
        return Tensor.of(Shape.of(b * n), f);
    }

    private Var lookup(Var table, int[][] v, int max) {
        return table.gather(0, ids(v, max)).reshape(v.length, v[0].length, dim);
    }

    /**
     * Encodes token sequences.
     *
     * @param tokens {@code [batch][n]} word ids
     * @param boxes {@code [batch][n][4]} boxes {@code x0, y0, x1, y1} in grid units
     * @param positions {@code [batch][n]} reading-order positions
     * @param training whether training
     * @return {@code [batch, n, dim]}
     */
    public Var encode(int[][] tokens, int[][][] boxes, int[][] positions, boolean training) {
        Var x = lookup(words, tokens, vocab - 1).add(lookup(order, positions, order.shape().dim(0) - 1));
        if (twoD) {
            int b = tokens.length;
            int n = tokens[0].length;
            int[][][] parts = new int[6][b][n];
            for (int i = 0; i < b; i++) {
                for (int t = 0; t < n; t++) {
                    int[] bx = boxes[i][t];
                    parts[0][i][t] = bx[0];
                    parts[1][i][t] = bx[2];
                    parts[2][i][t] = bx[1];
                    parts[3][i][t] = bx[3];
                    parts[4][i][t] = bx[2] - bx[0];
                    parts[5][i][t] = bx[3] - bx[1];
                }
            }
            x = x.add(lookup(xs, parts[0], grid - 1)).add(lookup(xs, parts[1], grid - 1))
                    .add(lookup(ys, parts[2], grid - 1)).add(lookup(ys, parts[3], grid - 1))
                    .add(lookup(ws, parts[4], grid - 1)).add(lookup(hs, parts[5], grid - 1));
        }
        for (Layer l : blocks) {
            x = l.forward(x, training);
        }
        return x;
    }

    /**
     * Tag emissions.
     *
     * @param tokens word ids
     * @param boxes boxes
     * @param positions reading-order positions
     * @param training whether training
     * @return {@code [batch, n, tags]}
     */
    public Var emissions(int[][] tokens, int[][][] boxes, int[][] positions, boolean training) {
        return encode(tokens, boxes, positions, training).matmul(tagW).add(tagB);
    }

    /**
     * The CRF tagging loss.
     *
     * @param tokens word ids
     * @param boxes boxes
     * @param positions reading-order positions
     * @param tags {@code [batch][n]} gold tags
     * @return the mean negative log-likelihood
     */
    public Var tagLoss(int[][] tokens, int[][][] boxes, int[][] positions, int[][] tags) {
        return crf.negativeLogLikelihood(emissions(tokens, boxes, positions, true), ids(tags, Integer.MAX_VALUE));
    }

    /**
     * Viterbi tags.
     *
     * @param tokens word ids
     * @param boxes boxes
     * @param positions reading-order positions
     * @return {@code [batch][n]} tags
     */
    public int[][] tag(int[][] tokens, int[][][] boxes, int[][] positions) {
        Tensor e = Var.noGrad(() -> emissions(tokens, boxes, positions, false)).value();
        return crf.decode(e);
    }

    /**
     * Masked visual-language modelling: mask a share of the tokens (keeping their boxes and
     * positions) and predict them.
     *
     * @param tokens word ids
     * @param boxes boxes
     * @param positions reading-order positions
     * @param share the share of tokens to mask
     * @param random the masking source
     * @return the cross-entropy over masked tokens
     */
    public Var maskedLoss(int[][] tokens, int[][][] boxes, int[][] positions, double share, SplittableRandom random) {
        int b = tokens.length;
        int n = tokens[0].length;
        int[][] masked = new int[b][n];
        float[] target = new float[b * n];
        boolean any = false;
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < n; t++) {
                boolean m = random.nextDouble() < share;
                masked[i][t] = m ? vocab - 1 : tokens[i][t];
                target[i * n + t] = m ? tokens[i][t] : -1;
                any |= m;
            }
        }
        if (!any) {
            masked[0][0] = vocab - 1;
            target[0] = tokens[0][0];
        }
        Var logits = encode(masked, boxes, positions, true).matmul(mlmW).add(mlmB).reshape(b * n, vocab);
        return Functional.crossEntropy(logits, Tensor.of(Shape.of(b * n), target));
    }

    /** @return the encoder, tag head and CRF parameters (and the masked-model head) */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(words, order));
        if (twoD) {
            p.addAll(List.of(xs, ys, ws, hs));
        }
        for (Layer l : blocks) {
            p.addAll(l.parameters());
        }
        p.addAll(List.of(tagW, tagB, mlmW, mlmB));
        p.addAll(crf.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "LayoutEncoder(" + (twoD ? "2-D" : "1-D") + " positions, " + blocks.size() + " blocks)";
    }
}
