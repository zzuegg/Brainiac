package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * StyleGAN-style residual block for sequences, conditioned on a style vector, as StyleTTS2
 * and Kokoro build their pitch, energy and decoder stacks from it: adaptive instance norm,
 * leaky ReLU, (optionally a learned 2x upsampling), a convolution, adaptive instance norm,
 * leaky ReLU, a convolution; the shortcut is upsampled by repetition and projected when the
 * width changes, and the sum is divided by {@code sqrt 2}.
 *
 * <p>Inputs are a sequence {@code [in, length]} and a style {@code [style]} per sample; the
 * output is {@code [out, length]}, or {@code [out, 2 length]} when upsampling. Wire it with
 * {@code Node.merge(AdainResBlock.of(out, style, false), style)}.
 */
public final class AdainResBlock implements MultiInputLayer {

    private final StyleNorm norm1;
    private final StyleNorm norm2;
    private final Var conv1W;
    private final Var conv1B;
    private final Var conv2W;
    private final Var conv2B;
    private final Var shortcutW;
    private final Var poolW;
    private final Var poolB;
    private final float slope;

    /**
     * A block with random weights.
     *
     * @param in the input channels
     * @param out the output channels
     * @param style the style width
     * @param upsample whether to double the length
     * @param random the random source
     */
    public AdainResBlock(int in, int out, int style, boolean upsample, SplittableRandom random) {
        this(new StyleNorm(StyleNorm.Kind.INSTANCE, in, style, random),
                new StyleNorm(StyleNorm.Kind.INSTANCE, out, style, random),
                conv(out, in, 3, random), Var.param(Tensor.zeros(Shape.of(out))),
                conv(out, out, 3, random), Var.param(Tensor.zeros(Shape.of(out))),
                in != out ? conv(out, in, 1, random) : null,
                upsample ? conv(in, 1, 3, random) : null, upsample ? Var.param(Tensor.zeros(Shape.of(in))) : null,
                0.2f);
    }

    /**
     * A block from given weights.
     *
     * @param norm1 the first norm, over the input channels
     * @param norm2 the second norm, over the output channels
     * @param conv1W the first convolution {@code [out, in, 3]}
     * @param conv1B its bias
     * @param conv2W the second convolution {@code [out, out, 3]}
     * @param conv2B its bias
     * @param shortcutW the shortcut projection {@code [out, in, 1]}, or {@code null} when {@code in == out}
     * @param poolW the depthwise transposed convolution that upsamples {@code [in, 1, 3]}, or
     *     {@code null} for no upsampling
     * @param poolB its bias
     * @param slope the leaky ReLU slope
     */
    public AdainResBlock(StyleNorm norm1, StyleNorm norm2, Var conv1W, Var conv1B, Var conv2W, Var conv2B,
            Var shortcutW, Var poolW, Var poolB, float slope) {
        this.norm1 = norm1;
        this.norm2 = norm2;
        this.conv1W = conv1W;
        this.conv1B = conv1B;
        this.conv2W = conv2W;
        this.conv2B = conv2B;
        this.shortcutW = shortcutW;
        this.poolW = poolW;
        this.poolB = poolB;
        this.slope = slope;
        if ((shortcutW == null) != (conv1W.shape().dim(0) == conv1W.shape().dim(1))) {
            throw new IllegalArgumentException("a shortcut projection is needed exactly when the width changes");
        }
    }

    static Var conv(int out, int in, int kernel, SplittableRandom random) {
        return Var.param(Init.HE.create(Shape.of(out, in, kernel), in * kernel, out * kernel, random));
    }

    /**
     * Describes a block.
     *
     * @param out the output channels
     * @param style the style width
     * @param upsample whether to double the length
     * @return the spec
     */
    public static MultiInputLayer.Spec of(int out, int style, boolean upsample) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                Shape s = StyleNorm.check(inputs, style);
                return Shape.of(out, upsample ? 2 * s.dim(1) : s.dim(1));
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new AdainResBlock(StyleNorm.check(inputs, style).dim(0), out, style, upsample, random);
            }
        };
    }

    /** @return whether the block doubles the length */
    public boolean upsamples() {
        return poolW != null;
    }

    /**
     * Runs the block.
     *
     * @param x {@code [batch, in, length]}
     * @param style {@code [batch, style]}
     * @return {@code [batch, out, length or 2 length]}
     */
    public Var forward(Var x, Var style) {
        Var h = norm1.forward(x, style).activation(ActivationOp.LEAKY_RELU, slope);
        if (poolW != null) {
            int c = poolW.shape().dim(0);
            h = Functional.convTranspose(h, poolW, poolB, new int[] {2}, new int[] {1}, new int[] {1}, new int[] {1},
                    new int[] {1}, c);
        }
        h = HifiGanBlock.conv(h, conv1W, conv1B, 1);
        h = norm2.forward(h, style).activation(ActivationOp.LEAKY_RELU, slope);
        h = HifiGanBlock.conv(h, conv2W, conv2B, 1);
        Var sc = poolW != null ? x.repeatInterleave(2, 2) : x;
        if (shortcutW != null) {
            sc = HifiGanBlock.conv(sc, shortcutW, null, 1);
        }
        return h.add(sc).mul((float) (1 / Math.sqrt(2)));
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException(this + " takes a sequence and a style, got " + inputs.size()
                    + " inputs");
        }
        return forward(inputs.get(0), inputs.get(1));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        Shape s = StyleNorm.check(inputs, norm1.parameters().get(0).shape().dim(0));
        return Shape.of(conv1W.shape().dim(0), poolW != null ? 2 * s.dim(1) : s.dim(1));
    }

    @Override
    public Shape outputShape(Shape input) {
        throw new IllegalStateException("an AdaIN block needs a style; wire it with Node.merge(spec, style)");
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(norm1.parameters());
        p.addAll(norm2.parameters());
        p.addAll(List.of(conv1W, conv1B, conv2W, conv2B));
        if (shortcutW != null) {
            p.add(shortcutW);
        }
        if (poolW != null) {
            p.addAll(List.of(poolW, poolB));
        }
        return p;
    }

    @Override
    public String toString() {
        return "AdainResBlock[" + conv1W.shape().dim(1) + " -> " + conv1W.shape().dim(0) + (upsamples() ? ", x2" : "")
                + "]";
    }

    static Config write(AdainResBlock b) {
        return Config.builder().put("in", b.conv1W.shape().dim(1)).put("out", b.conv1W.shape().dim(0))
                .put("style", b.norm1.parameters().get(0).shape().dim(0)).put("upsample", b.upsamples())
                .put("slope", b.slope).build();
    }

    static AdainResBlock read(Config c) {
        AdainResBlock r = new AdainResBlock(c.getInt("in"), c.getInt("out"), c.getInt("style"),
                c.getBoolean("upsample"), new SplittableRandom());
        return new AdainResBlock(r.norm1, r.norm2, r.conv1W, r.conv1B, r.conv2W, r.conv2B, r.shortcutW, r.poolW,
                r.poolB, c.getFloat("slope"));
    }
}
