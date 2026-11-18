package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A ConvNeXt block over sequences (Liu et al., 2022), as Vocos and F5-TTS use them on audio
 * frames and text: a depthwise convolution (kernel 7) mixes along the sequence, layer norm
 * over the channels, then an inverted-bottleneck MLP per position ({@code channels -> hidden
 * -> channels}, exact GELU) and a residual connection. Two optional parts:
 * <ul>
 * <li>global response normalization (GRN, ConvNeXt V2, Woo et al., 2023) after the GELU:
 *     {@code gamma * (x * N) + beta + x}, with {@code N} each hidden feature's L2 norm over
 *     the sequence divided by the mean of those norms, so features compete;</li>
 * <li>a learned per-channel layer scale on the branch (ConvNeXt V1, as Vocos).</li>
 * </ul>
 *
 * <p>Input and output are {@code [channels, length]}, padded so the length stays.
 */
public final class ConvNeXt1d implements Layer {

    private final float eps;
    private final Var depthwise;
    private final Var depthwiseBias;
    private final Var normWeight;
    private final Var normBias;
    private final Var up;
    private final Var upBias;
    private final Var grnGamma;
    private final Var grnBeta;
    private final Var down;
    private final Var downBias;
    private final Var scale;

    /**
     * A block from given weights.
     *
     * @param eps the layer norm's epsilon (1e-6 in Vocos and F5-TTS)
     * @param depthwise {@code [channels, 1, kernel]}, odd kernel
     * @param depthwiseBias {@code [channels]}
     * @param normWeight {@code [channels]}
     * @param normBias {@code [channels]}
     * @param up {@code [channels, hidden]}
     * @param upBias {@code [hidden]}
     * @param grnGamma {@code [hidden]}, or {@code null} for no GRN
     * @param grnBeta {@code [hidden]}, or {@code null}
     * @param down {@code [hidden, channels]}
     * @param downBias {@code [channels]}
     * @param scale the layer scale {@code [channels]}, or {@code null}
     */
    public ConvNeXt1d(float eps, Var depthwise, Var depthwiseBias, Var normWeight, Var normBias, Var up, Var upBias,
            Var grnGamma, Var grnBeta, Var down, Var downBias, Var scale) {
        if (depthwise.shape().rank() != 3 || depthwise.shape().dim(1) != 1 || depthwise.shape().dim(2) % 2 == 0) {
            throw new IllegalArgumentException("a ConvNeXt depthwise kernel is [channels, 1, odd kernel], got "
                    + depthwise.shape());
        }
        if ((grnGamma == null) != (grnBeta == null)) {
            throw new IllegalArgumentException("GRN needs both gamma and beta");
        }
        this.eps = eps;
        this.depthwise = depthwise;
        this.depthwiseBias = depthwiseBias;
        this.normWeight = normWeight;
        this.normBias = normBias;
        this.up = up;
        this.upBias = upBias;
        this.grnGamma = grnGamma;
        this.grnBeta = grnBeta;
        this.down = down;
        this.downBias = downBias;
        this.scale = scale;
    }

    /**
     * A block with random weights.
     *
     * @param channels the channels
     * @param kernel the depthwise kernel, odd
     * @param hidden the MLP's width
     * @param grn whether to add global response normalization
     * @param layerScale the layer scale's start value, or 0 for none
     * @param random the random source
     */
    public ConvNeXt1d(int channels, int kernel, int hidden, boolean grn, float layerScale, SplittableRandom random) {
        this(1e-6f, Var.param(Init.HE.create(Shape.of(channels, 1, kernel), kernel, kernel, random)),
                zeros(channels), Var.param(Tensor.ones(Shape.of(channels))), zeros(channels),
                Var.param(Init.XAVIER.create(Shape.of(channels, hidden), channels, hidden, random)), zeros(hidden),
                grn ? zeros(hidden) : null, grn ? zeros(hidden) : null,
                Var.param(Init.XAVIER.create(Shape.of(hidden, channels), hidden, channels, random)), zeros(channels),
                layerScale > 0 ? Var.param(Tensor.full(Shape.of(channels), layerScale)) : null);
    }

    private static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    /**
     * Describes a block for a model on {@code [channels, length]}.
     *
     * @param kernel the depthwise kernel, odd (7)
     * @param expansion the MLP's width over the channels (3 in Vocos, 2 in F5-TTS, 4 in ConvNeXt)
     * @param grn whether to add global response normalization (ConvNeXt V2)
     * @param layerScale the layer scale's start value (such as 1e-6), or 0 for none
     * @return the spec
     */
    public static LayerSpec of(int kernel, int expansion, boolean grn, float layerScale) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                check(input);
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                check(input);
                return new ConvNeXt1d(input.dim(0), kernel, expansion * input.dim(0), grn, layerScale, random);
            }
        };
    }

    private static void check(Shape input) {
        if (input.rank() != 2) {
            throw new IllegalArgumentException("a ConvNeXt block takes [channels, length], got " + input);
        }
    }

    /** @return the channels */
    public int channels() {
        return depthwise.shape().dim(0);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(1) != channels()) {
            throw new IllegalArgumentException(this + " expects [batch, " + channels() + ", length], got " + s);
        }
        int c = channels();
        int k = depthwise.shape().dim(2);
        int p = k / 2;
        Var h = Functional.conv(input, depthwise, depthwiseBias, new int[] {1}, new int[] {p}, new int[] {p},
                new int[] {1}, c).permute(0, 2, 1);                    // [batch, length, channels]
        Var centred = h.sub(h.mean(2, true));
        h = centred.div(centred.square().mean(2, true).add(eps).sqrt()).mul(normWeight).add(normBias);
        h = Activation.GELU.forward(h.matmul(up).add(upBias));
        if (grnGamma != null) {
            Var g = h.square().sum(1, true).sqrt();                     // [batch, 1, hidden]
            Var n = g.div(g.mean(2, true).add(1e-6f));
            h = h.mul(n).mul(grnGamma).add(grnBeta).add(h);
        }
        h = h.matmul(down).add(downBias);
        if (scale != null) {
            h = h.mul(scale);
        }
        return input.add(h.permute(0, 2, 1));
    }

    @Override
    public Shape outputShape(Shape input) {
        check(input);
        return input;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(depthwise, depthwiseBias, normWeight, normBias, up, upBias));
        if (grnGamma != null) {
            p.add(grnGamma);
            p.add(grnBeta);
        }
        p.add(down);
        p.add(downBias);
        if (scale != null) {
            p.add(scale);
        }
        return p;
    }

    @Override
    public String toString() {
        return "ConvNeXt1d[" + channels() + " channels, kernel " + depthwise.shape().dim(2) + ", hidden "
                + up.shape().dim(1) + (grnGamma != null ? ", GRN" : "") + (scale != null ? ", layer scale" : "") + "]";
    }

    static Config write(ConvNeXt1d b) {
        return Config.builder().put("channels", b.channels()).put("kernel", b.depthwise.shape().dim(2))
                .put("hidden", b.up.shape().dim(1)).put("grn", b.grnGamma != null).put("layerScale", b.scale != null)
                .put("eps", b.eps).build();
    }

    static ConvNeXt1d read(Config c) {
        int ch = c.getInt("channels");
        int h = c.getInt("hidden");
        boolean grn = c.getBoolean("grn");
        return new ConvNeXt1d(c.getFloat("eps"), Var.param(Tensor.zeros(Shape.of(ch, 1, c.getInt("kernel")))),
                zeros(ch), zeros(ch), zeros(ch), Var.param(Tensor.zeros(Shape.of(ch, h))), zeros(h),
                grn ? zeros(h) : null, grn ? zeros(h) : null, Var.param(Tensor.zeros(Shape.of(h, ch))), zeros(ch),
                c.getBoolean("layerScale") ? zeros(ch) : null);
    }
}
