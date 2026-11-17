package io.github.zzuegg.brainiac.experimental.nn;

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
 * as a {@link Denoiser} for small single-channel images. The image is cut into
 * {@code patch x patch} patches, each a token; a stack of transformer blocks mixes them;
 * a final linear layer turns every token back into its patch.
 *
 * <p>Conditioning on the time and the class uses <em>adaLN-Zero</em>: a conditioning vector
 * {@code c = MLP(time features) + classEmbedding} is mapped, per block, to a shift, a scale
 * and a gate for each of the two sub-layers:
 * {@code x += gate * Attention(LayerNorm(x) * (1 + scale) + shift)} (and the same for the
 * MLP). Those maps start at exactly zero, so every block starts as the identity and the
 * whole network starts by predicting 0: the property that made DiT train stably, and that
 * {@link #forward} shows on a fresh network.
 *
 * <pre>{@code
 * Denoiser dit = DiffusionTransformer.builder(8, 8).patch(2).width(32).depth(2).heads(2).classes(4).build();
 * }</pre>
 */
public final class DiffusionTransformer implements Denoiser {
    private static final int TIME_FEATURES = 32;
    private final int height;
    private final int width;
    private final int patch;
    private final int dim;
    private final int heads;
    private final int classes;
    private final int tokens;
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
            int b = x.shape().dim(0);
            Var m = modulation.forward(GenBlocks.silu(c));   // [b, 6 dim]
            Var h = modulate(layerNorm(x), slice(m, 0, b), slice(m, 1, b));
            x = x.add(attention(h).mul(slice(m, 2, b)));
            h = modulate(layerNorm(x), slice(m, 3, b), slice(m, 4, b));
            return x.add(down.forward(up.forward(h).activation(io.github.zzuegg.brainiac.tensor.spi.ActivationOp.GELU, 0))
                    .mul(slice(m, 5, b)));
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

    /** The i-th {@code dim}-wide chunk of a modulation vector, as {@code [b, 1, dim]}. */
    private Var slice(Var m, int i, int b) {
        return m.slice(1, i * dim, (i + 1) * dim).reshape(b, 1, dim);
    }

    private static Var modulate(Var x, Var shift, Var scale) {
        return x.mul(scale.add(1f)).add(shift);
    }

    /** LayerNorm over the last axis without affine parameters (adaLN supplies them). */
    private static Var layerNorm(Var x) {
        int last = x.shape().rank() - 1;
        Var centred = x.sub(x.mean(last, true));
        return centred.div(centred.square().mean(last, true).add(1e-6f).sqrt());
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
    public Var forward(Var x, float[] t, int[] labels) {
        int b = x.shape().dim(0);
        Var h = embed.forward(patchify(x)).add(position);
        Tensor oneHot = GenBlocks.oneHot(labels, classes).to(x.value().device());
        Var c = time2.forward(GenBlocks.silu(time1.forward(Var.of(GenBlocks.timeEmbedding(t, TIME_FEATURES)
                .to(x.value().device()))))).add(Var.of(oneHot).matmul(classTable));
        for (Block block : blocks) {
            h = block.forward(h, c);
        }
        Var m = finalModulation.forward(GenBlocks.silu(c));
        h = modulate(layerNorm(h), slice(m, 0, b), slice(m, 1, b));
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
