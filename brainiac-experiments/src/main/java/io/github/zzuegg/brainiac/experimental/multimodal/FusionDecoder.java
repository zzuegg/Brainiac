package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A small decoder-only language model whose input is a sequence of <em>embeddings</em>
 * rather than token ids, so continuous tokens (projected image patches, perceiver latents,
 * diffusion patches) can be placed among the text, and whose residual stream can be edited
 * between blocks (gated cross-attention layers). Core {@link TransformerBlock}s run with an
 * explicit attention bias: causal by default, or any mask, such as Transfusion's causal mask
 * with bidirectional image spans.
 *
 * <p>It is the frozen language model of the vision-language bridges in this package and the
 * shared trunk of {@link TransfusionModel}.
 */
public final class FusionDecoder {

    /** Edits the residual stream after a block. */
    @FunctionalInterface
    public interface BlockHook {

        /**
         * Called after each block.
         *
         * @param block the block index
         * @param x the residual stream {@code [batch, steps, dim]}
         * @return the stream later blocks see
         */
        Var after(int block, Var x);
    }

    private final int vocab;
    private final int dim;
    private final int maxLength;
    private final Var tokens;
    private final Var positions;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final LayerNorm norm;
    private final Var head;
    private final Var headBias;

    /**
     * Creates the decoder.
     *
     * @param vocab the vocabulary
     * @param dim the width
     * @param layers blocks
     * @param heads attention heads
     * @param maxLength the longest sequence (learned positions)
     * @param seed the seed
     */
    public FusionDecoder(int vocab, int dim, int layers, int heads, int maxLength, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.vocab = vocab;
        this.dim = dim;
        this.maxLength = maxLength;
        this.tokens = Var.param(Tensor.normal(Shape.of(vocab, dim), 0, 0.3f, r));
        this.positions = Var.param(Tensor.normal(Shape.of(maxLength, dim), 0, 0.1f, r));
        for (int i = 0; i < layers; i++) {
            blocks.add((TransformerBlock) TransformerBlock.of(heads).ffnSize(2 * dim).create(Shape.of(8, dim), r));
        }
        this.norm = LayerNorm.of(dim);
        this.head = Var.param(Tensor.normal(Shape.of(dim, vocab), 0, (float) Math.sqrt(1.0 / dim), r));
        this.headBias = Var.param(Tensor.zeros(Shape.of(vocab)));
    }

    /** @return the width */
    public int dim() {
        return dim;
    }

    /** @return the vocabulary size */
    public int vocab() {
        return vocab;
    }

    /** @return the number of blocks */
    public int layers() {
        return blocks.size();
    }

    /**
     * Token embeddings (without positions).
     *
     * @param ids {@code [batch][steps]}
     * @return {@code [batch, steps, dim]}
     */
    public Var embed(int[][] ids) {
        int b = ids.length;
        int n = ids[0].length;
        float[] f = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < n; t++) {
                f[i * n + t] = ids[i][t];
            }
        }
        return tokens.gather(0, Tensor.of(Shape.of(b * n), f)).reshape(b, n, dim);
    }

    /**
     * A causal attention bias.
     *
     * @param steps the sequence length
     * @return {@code [1, 1, steps, steps]} with {@code -1e9} above the diagonal
     */
    public static Tensor causal(int steps) {
        float[] m = new float[steps * steps];
        for (int i = 0; i < steps; i++) {
            for (int j = i + 1; j < steps; j++) {
                m[i * steps + j] = -1e9f;
            }
        }
        return Tensor.of(Shape.of(1, 1, steps, steps), m);
    }

    /**
     * Runs the blocks on embeddings, with learned positions added.
     *
     * @param x {@code [batch, steps, dim]}
     * @param bias the attention bias, broadcast to {@code [batch, heads, steps, steps]}; null for causal
     * @param hook edits after each block, or null
     * @param training whether training
     * @return the final hidden states {@code [batch, steps, dim]} (normalised)
     */
    public Var hidden(Var x, Tensor bias, BlockHook hook, boolean training) {
        int steps = x.shape().dim(1);
        if (steps > maxLength) {
            throw new IllegalArgumentException("sequence of " + steps + " longer than " + maxLength);
        }
        Tensor mask = bias == null ? causal(steps) : bias;
        Var h = x.add(positions.slice(0, 0, steps));
        for (int i = 0; i < blocks.size(); i++) {
            h = blocks.get(i).forward(List.of(h), mask, training);
            if (hook != null) {
                h = hook.after(i, h);
            }
        }
        return norm.forward(h);
    }

    /**
     * Next-token logits.
     *
     * @param x embeddings {@code [batch, steps, dim]}
     * @param bias the attention bias or null for causal
     * @param hook edits after each block, or null
     * @param training whether training
     * @return {@code [batch, steps, vocab]}
     */
    public Var logits(Var x, Tensor bias, BlockHook hook, boolean training) {
        return project(hidden(x, bias, hook, training));
    }

    /**
     * The output head on hidden states.
     *
     * @param hidden {@code [..., dim]} final hidden states
     * @return the logits {@code [..., vocab]}
     */
    public Var project(Var hidden) {
        return hidden.matmul(head).add(headBias);
    }

    /**
     * Cross-entropy of logits against targets, skipping negative targets.
     *
     * @param logits {@code [batch, steps, vocab]}
     * @param targets {@code [batch][steps]}, -1 to skip
     * @return the mean loss over the counted targets
     */
    public static Var loss(Var logits, int[][] targets) {
        int b = targets.length;
        int n = targets[0].length;
        int v = logits.shape().dim(2);
        float[] t = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < n; k++) {
                t[i * n + k] = targets[i][k];
            }
        }
        return Functional.crossEntropy(logits.reshape(b * n, v), Tensor.of(Shape.of(b * n), t));
    }

    /**
     * Freezes or unfreezes nothing by itself; this returns every trainable value so callers
     * can leave them out of an optimizer.
     *
     * @return the parameters
     */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(tokens, positions));
        for (TransformerBlock blk : blocks) {
            p.addAll(blk.parameters());
        }
        p.addAll(norm.parameters());
        p.add(head);
        p.add(headBias);
        return p;
    }

    /** Clears the gradients a frozen decoder accumulates when bridges are trained through it. */
    public void zeroGrad() {
        for (Var p : parameters()) {
            p.zeroGrad();
        }
    }

    @Override
    public String toString() {
        return "FusionDecoder(" + blocks.size() + " blocks, " + dim + " wide)";
    }
}
