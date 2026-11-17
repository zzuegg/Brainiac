package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A Diffusion Transformer (DiT, Peebles and Xie, <a href="https://arxiv.org/abs/2212.09748">2212.09748</a>)
 * as a {@link ConditionalDenoiser} for small single-channel images. The image is cut into
 * {@code patch x patch} patches, each a token; a stack of transformer blocks mixes them;
 * a final linear layer turns every token back into its patch.
 *
 * <p>Conditioning on the time and the class uses <em>adaLN-Zero</em>: a conditioning vector
 * {@code c = MLP(time features) + classEmbedding} is mapped, per block, to a shift, a scale
 * and a gate for each of the two sub-layers:
 * {@code x += gate * Attention(LayerNorm(x) * (1 + scale) + shift)} (and the same for the
 * MLP). Those maps start at exactly zero, so every block starts as the identity and the
 * whole network starts by predicting 0: the property that made DiT train stably, and that
 * {@link #apply} shows on a fresh network. The modulation and the gates are the core
 * {@code Functional.adaLayerNorm} and {@code Functional.gate}.
 *
 * <pre>{@code
 * DiffusionTransformer dit = DiffusionTransformer.builder(8, 8).patch(2).width(32).depth(2).heads(2).classes(4).build();
 * }</pre>
 */
public final class DiffusionTransformer implements ConditionalDenoiser {
    private static final int TIME_FEATURES = 32;
    private static final float EPS = 1e-6f;
    private final int height;
    private final int width;
    private final int patch;
    private final int dim;
    private final int heads;
    private final int classes;
    private final int tokens;
    private final float timeScale;
    private final Dense embed;
    private final Var position;
    private final Dense time1;
    private final Dense time2;
    private final Var classTable;
    private final List<Block> blocks = new ArrayList<>();
    private final Dense finalModulation;
    private final Dense finalLinear;

    private DiffusionTransformer(Builder b) {
        SplittableRandom random = new SplittableRandom(b.seed);
        if (b.height % b.patch != 0 || b.width % b.patch != 0 || b.dim % b.heads != 0) {
            throw new IllegalArgumentException("the patch must divide the image and the heads the width");
        }
        height = b.height;
        width = b.width;
        patch = b.patch;
        dim = b.dim;
        heads = b.heads;
        classes = b.classes;
        tokens = (height / patch) * (width / patch);
        timeScale = b.timeScale;
        embed = GenBlocks.linear(patch * patch, dim, random);
        position = Var.param(Tensor.normal(Shape.of(tokens, dim), 0, 0.02f, random));
        time1 = GenBlocks.linear(TIME_FEATURES, dim, random);
        time2 = GenBlocks.linear(dim, dim, random);
        classTable = Var.param(Tensor.normal(Shape.of(classes + 1, dim), 0, 0.5f, random));
        for (int i = 0; i < b.depth; i++) {
            blocks.add(new Block(random));
        }
        finalModulation = GenBlocks.zeroLinear(dim, 2 * dim);
        finalLinear = GenBlocks.zeroLinear(dim, patch * patch);
    }

    /**
     * Starts a builder.
     *
     * @param height image height
     * @param width image width
     * @return the builder
     */
    public static Builder builder(int height, int width) {
        return new Builder(height, width);
    }

    /** Settings for a {@link DiffusionTransformer}. */
    public static final class Builder {
        private final int height;
        private final int width;
        private int patch = 2;
        private int dim = 32;
        private int depth = 2;
        private int heads = 2;
        private int classes;
        private long seed = 42;
        private float timeScale = 1;

        private Builder(int height, int width) {
            this.height = height;
            this.width = width;
        }

        /** @param p the patch side @return this builder */
        public Builder patch(int p) {
            this.patch = p;
            return this;
        }

        /** @param d the token width @return this builder */
        public Builder width(int d) {
            this.dim = d;
            return this;
        }

        /** @param n the number of blocks @return this builder */
        public Builder depth(int n) {
            this.depth = n;
            return this;
        }

        /** @param n the attention heads @return this builder */
        public Builder heads(int n) {
            this.heads = n;
            return this;
        }

        /** @param n the number of classes, 0 for unconditional @return this builder */
        public Builder classes(int n) {
            this.classes = n;
            return this;
        }

        /**
         * Sets the factor applied to incoming times: {@code 1 / steps} for core
         * {@code Diffusion}, 1 (the default) for flow matching.
         *
         * @param scale the factor
         * @return this builder
         */
        public Builder timeScale(float scale) {
            this.timeScale = scale;
            return this;
        }

        /** @param s the initialization seed @return this builder */
        public Builder seed(long s) {
            this.seed = s;
            return this;
        }

        /** @return the network */
        public DiffusionTransformer build() {
            return new DiffusionTransformer(this);
        }
    }

    /** One DiT block with adaLN-Zero modulation. */
    private final class Block {
        final Dense modulation;
        final Dense qkv;
        final Dense proj;
        final Dense up;
        final Dense down;

        Block(SplittableRandom random) {
            modulation = GenBlocks.zeroLinear(dim, 6 * dim);
            qkv = GenBlocks.linear(dim, 3 * dim, random);
            proj = GenBlocks.linear(dim, dim, random);
            up = GenBlocks.linear(dim, 4 * dim, random);
            down = GenBlocks.linear(4 * dim, dim, random);
        }

        Var forward(Var x, Var c) {
            Var m = modulation.forward(GenBlocks.silu(c));   // [b, 6 dim]: shift, scale, gate twice
            Var h = Functional.adaLayerNorm(x, chunk(m, 1), chunk(m, 0), EPS);
            x = Functional.gate(x, attention(h), chunk(m, 2));
            h = Functional.adaLayerNorm(x, chunk(m, 4), chunk(m, 3), EPS);
            return Functional.gate(x, down.forward(up.forward(h)
                    .activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.GELU, 0)), chunk(m, 5));
        }

        Var attention(Var h) {
            int b = h.shape().dim(0);
            int dh = dim / heads;
            Var all = qkv.forward(h);   // [b, n, 3 dim]
            Var q = all.slice(2, 0, dim).reshape(b, tokens, heads, dh).permute(0, 2, 1, 3);
            Var k = all.slice(2, dim, 2 * dim).reshape(b, tokens, heads, dh).permute(0, 2, 1, 3);
            Var v = all.slice(2, 2 * dim, 3 * dim).reshape(b, tokens, heads, dh).permute(0, 2, 1, 3);
            Var o = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(dh)));
            return proj.forward(o.permute(0, 2, 1, 3).reshape(b, tokens, dim));
        }
    }

    /** The i-th {@code dim}-wide chunk of a modulation vector, {@code [b, dim]}. */
    private Var chunk(Var m, int i) {
        return m.slice(1, i * dim, (i + 1) * dim);
    }

    /** Cuts {@code [b, h*w]} images into {@code [b, tokens, patch*patch]}. */
    Var patchify(Var x) {
        int b = x.shape().dim(0);
        int ph = height / patch;
        int pw = width / patch;
        return x.reshape(b, ph, patch, pw, patch).permute(0, 1, 3, 2, 4).reshape(b, tokens, patch * patch);
    }

    /** The inverse of {@link #patchify}. */
    Var unpatchify(Var x) {
        int b = x.shape().dim(0);
        int ph = height / patch;
        int pw = width / patch;
        return x.reshape(b, ph, pw, patch, patch).permute(0, 1, 3, 2, 4).reshape(b, height * width);
    }

    @Override
    public Var apply(Var x, Tensor time, int[] labels) {
        float[] t = time.toArray();
        for (int i = 0; i < t.length; i++) {
            t[i] *= timeScale;
        }
        Var h = embed.forward(patchify(x)).add(position);
        Tensor oneHot = GenBlocks.oneHot(labels, classes).to(x.value().device());
        Var c = time2.forward(GenBlocks.silu(time1.forward(Var.of(GenBlocks.timeEmbedding(t, TIME_FEATURES)
                .to(x.value().device()))))).add(Var.of(oneHot).matmul(classTable));
        for (Block block : blocks) {
            h = block.forward(h, c);
        }
        Var m = finalModulation.forward(GenBlocks.silu(c));
        h = Functional.adaLayerNorm(h, chunk(m, 1), chunk(m, 0), EPS);
        return unpatchify(finalLinear.forward(h));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(embed.parameters());
        p.add(position);
        p.addAll(time1.parameters());
        p.addAll(time2.parameters());
        p.add(classTable);
        for (Block block : blocks) {
            p.addAll(block.modulation.parameters());
            p.addAll(block.qkv.parameters());
            p.addAll(block.proj.parameters());
            p.addAll(block.up.parameters());
            p.addAll(block.down.parameters());
        }
        p.addAll(finalModulation.parameters());
        p.addAll(finalLinear.parameters());
        return p;
    }

    @Override
    public int dimension() {
        return height * width;
    }

    @Override
    public int classes() {
        return classes;
    }
}
