package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Multi-head self-attention over {@code [batch, steps, features]} with a choice of position
 * encoding and optional QK-norm, so position schemes can be compared in one otherwise
 * identical layer (a drop-in for {@code MultiHeadAttention.of(heads)} in sequential models).
 *
 * <p><b>QK-norm</b> RMS-normalizes queries and keys per head (with a learned gain) before
 * the dot product. Scores then stay bounded, which removes the attention-logit growth that
 * makes large or high-learning-rate transformers diverge; used by ViT-22B, OLMo 2, Gemma 3
 * and Qwen 3. Dehghani et al., "Scaling Vision Transformers to 22 Billion Parameters",
 * <a href="https://arxiv.org/abs/2302.05442">arXiv:2302.05442</a>; Henry et al.,
 * <a href="https://arxiv.org/abs/2010.04245">arXiv:2010.04245</a>.
 *
 * <p>Position encodings: see {@link Position} and {@link PositionEncodings}.
 */
public final class PositionalAttention implements Layer {

    /** Where position information comes from. */
    public enum Position {
        /** NoPE: none; with a causal mask the model can still infer order. */
        NONE,
        /** Rotary embedding (RoPE) of queries and keys. */
        ROTARY,
        /** xPos: RoPE with an exponential distance decay. */
        XPOS,
        /** ALiBi: fixed linear distance penalties per head. */
        ALIBI,
        /** T5: a learned bias per head and log-spaced distance bucket. */
        T5,
        /** CoPE: learned embeddings at contextual (gate-counted) positions. */
        COPE,
        /** 2-D RoPE over a patch grid; set the grid with {@link Settings#grid}. */
        ROTARY_2D
    }

    /**
     * The layer's settings.
     *
     * @param features the model width
     * @param heads the number of heads, dividing {@code features}
     * @param causal whether each step sees only itself and earlier steps
     * @param position the position encoding
     * @param qkNorm whether to RMS-normalize queries and keys per head
     * @param rotaryBase the base for the rotary variants
     * @param buckets T5 buckets
     * @param maxDistance T5 distance of the last bucket
     * @param copePositions the number of CoPE position embeddings
     * @param gridHeight 2-D RoPE patch rows
     * @param gridWidth 2-D RoPE patch columns
     * @param prefix 2-D RoPE leading non-patch tokens (such as a class token)
     */
    public record Settings(int features, int heads, boolean causal, Position position, boolean qkNorm,
            float rotaryBase, int buckets, int maxDistance, int copePositions, int gridHeight, int gridWidth,
            int prefix) {

        /**
         * Default settings: rotary base 10000, 32 T5 buckets up to distance 128, 16 CoPE
         * positions, no grid.
         *
         * @param features the model width
         * @param heads the number of heads
         * @param causal whether attention is causal
         * @param position the position encoding
         * @param qkNorm whether to use QK-norm
         * @return the settings
         */
        public static Settings of(int features, int heads, boolean causal, Position position, boolean qkNorm) {
            return new Settings(features, heads, causal, position, qkNorm, 10000f, 32, 128, 16, 0, 0, 0);
        }

        /**
         * Sets the 2-D RoPE patch grid (and its rotary base 100).
         *
         * @param height patch rows
         * @param width patch columns
         * @param prefixTokens leading tokens that are not patches
         * @return new settings
         */
        public Settings grid(int height, int width, int prefixTokens) {
            return new Settings(features, heads, causal, position, qkNorm, 100f, buckets, maxDistance,
                    copePositions, height, width, prefixTokens);
        }
    }

    private final Settings s;
    private final int headSize;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var qGain;
    private final Var kGain;
    private final Var t5Table;
    private final Var copeEmbeddings;

    /**
     * Creates the layer.
     *
     * @param settings the settings
     * @param random the random source
     */
    public PositionalAttention(Settings settings, RandomGenerator random) {
        this.s = settings;
        if (s.features() % s.heads() != 0) {
            throw new IllegalArgumentException(s.heads() + " heads do not divide " + s.features() + " features");
        }
        this.headSize = s.features() / s.heads();
        int d = s.features();
        wq = TechniqueOps.weight(random, d, d);
        wk = TechniqueOps.weight(random, d, d);
        wv = TechniqueOps.weight(random, d, d);
        wo = TechniqueOps.weight(random, d, d);
        qGain = s.qkNorm() ? Var.param(Tensor.ones(Shape.of(headSize))) : null;
        kGain = s.qkNorm() ? Var.param(Tensor.ones(Shape.of(headSize))) : null;
        t5Table = s.position() == Position.T5 ? Var.param(Tensor.zeros(Shape.of(s.heads(), s.buckets()))) : null;
        copeEmbeddings = s.position() == Position.COPE
                ? TechniqueOps.normal(random, (float) (1 / Math.sqrt(headSize)), s.copePositions(), headSize) : null;
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}, reading the width from
     * the input {@code [steps, features]}.
     *
     * @param heads the number of heads
     * @param causal whether attention is causal
     * @param position the position encoding
     * @param qkNorm whether to use QK-norm
     * @return the spec
     */
    public static LayerSpec spec(int heads, boolean causal, Position position, boolean qkNorm) {
        return TechniqueOps.sameShape((Shape in, SplittableRandom r) -> new PositionalAttention(
                Settings.of(in.dim(in.rank() - 1), heads, causal, position, qkNorm), r));
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 3 || input.shape().dim(2) != s.features()) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + s.features() + "], got "
                    + input.shape());
        }
        int t = input.shape().dim(1);
        Var q = TechniqueOps.splitHeads(input.matmul(wq), s.heads());
        Var k = TechniqueOps.splitHeads(input.matmul(wk), s.heads());
        Var v = TechniqueOps.splitHeads(input.matmul(wv), s.heads());
        if (s.qkNorm()) {
            q = TechniqueOps.rms(q, 1e-6f).mul(qGain);
            k = TechniqueOps.rms(k, 1e-6f).mul(kGain);
        }
        Var bias = null;
        Tensor mask = s.causal() ? TechniqueOps.causalMask(t, input.value().device()) : null;
        switch (s.position()) {
            case ROTARY -> {
                q = Functional.rotary(q, s.rotaryBase(), 0);
                k = Functional.rotary(k, s.rotaryBase(), 0);
            }
            case XPOS -> {
                q = PositionEncodings.xpos(q, s.rotaryBase(), 512f, 0, true);
                k = PositionEncodings.xpos(k, s.rotaryBase(), 512f, 0, false);
            }
            case ROTARY_2D -> {
                q = PositionEncodings.rotary2d(q, s.gridHeight(), s.gridWidth(), s.rotaryBase(), s.prefix());
                k = PositionEncodings.rotary2d(k, s.gridHeight(), s.gridWidth(), s.rotaryBase(), s.prefix());
            }
            case ALIBI -> {
                Tensor alibi = Functional.alibi(s.heads(), t, t, input.value().device());
                mask = mask == null ? alibi : alibi.add(mask);
            }
            case T5 -> bias = PositionEncodings.t5Bias(t5Table, t, t, !s.causal(), s.maxDistance());
            case COPE -> bias = PositionEncodings.cope(q, k, copeEmbeddings, s.causal());
            case NONE -> {
            }
        }
        Var o = TechniqueOps.attend(q, k, v, bias, mask, (float) (1 / Math.sqrt(headSize)));
        return TechniqueOps.mergeHeads(o).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, wo));
        if (qGain != null) {
            p.add(qGain);
            p.add(kGain);
        }
        if (t5Table != null) {
            p.add(t5Table);
        }
        if (copeEmbeddings != null) {
            p.add(copeEmbeddings);
        }
        return p;
    }

    /** @return the value projection {@code [features, features]} */
    public Var valueWeight() {
        return wv;
    }

    /** @return the output projection {@code [features, features]} */
    public Var outputWeight() {
        return wo;
    }

    /** @return the settings */
    public Settings settings() {
        return s;
    }

    @Override
    public String toString() {
        return "PositionalAttention(" + s.features() + ", " + s.heads() + " heads, " + s.position()
                + (s.qkNorm() ? ", qk-norm" : "") + (s.causal() ? ", causal" : "") + ")";
    }
}
