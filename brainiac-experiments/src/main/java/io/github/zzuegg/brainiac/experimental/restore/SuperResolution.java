package io.github.zzuegg.brainiac.experimental.restore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.PixelShuffle;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Single-image super-resolution networks, {@code [batch, 1, h, w] -> [batch, 1, h * r, w * r]}:
 *
 * <ul>
 *   <li>{@link Kind#SRCNN} (Dong et al., ECCV 2014): upsample bicubically first, then refine
 *       with three convolutions (9x9 patch extraction, 1x1 non-linear mapping, 5x5
 *       reconstruction), all at high resolution, so it is the most expensive per pixel;</li>
 *   <li>{@link Kind#ESPCN} (Shi et al., CVPR 2016): every convolution at low resolution, then a
 *       sub-pixel convolution (a convolution to {@code r^2} channels and core
 *       {@link PixelShuffle}), about {@code r^2} times cheaper;</li>
 *   <li>{@link Kind#EDSR} (Lim et al., CVPRW 2017): residual blocks without batch norm and with
 *       residual scaling, a long skip around the body, then the sub-pixel upsampler;</li>
 *   <li>{@link Kind#SPAN} (Wan et al., 2024): ESPCN-sized blocks with a parameter-free
 *       attention: each block's third convolution output {@code h} gives the map
 *       {@code sigmoid(h) - 0.5}, odd and origin-symmetric, and the block returns
 *       {@code (h + x) * (sigmoid(h) - 0.5)}, so strongly negative features also count.</li>
 * </ul>
 * Every variant adds its output to the bicubic upsampling of the input (a global residual, as
 * VDSR and SPAN's input skip do) so that short trainings start at the bicubic baseline; with
 * {@code globalResidual = false} the networks are as in their papers. {@link #parameterCount()}
 * and {@link #macs(int, int)} give the cost for a low-resolution input size.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1501.00092">arXiv:1501.00092</a>,
 * <a href="https://arxiv.org/abs/1609.05158">arXiv:1609.05158</a>,
 * <a href="https://arxiv.org/abs/1707.02921">arXiv:1707.02921</a>,
 * <a href="https://arxiv.org/abs/2311.12770">arXiv:2311.12770</a>.
 */
public final class SuperResolution implements Layer {

    /** The architecture. */
    public enum Kind {
        /** Bicubic upsampling, then three convolutions at high resolution. */
        SRCNN,
        /** Low-resolution convolutions and a sub-pixel upsampler. */
        ESPCN,
        /** Residual blocks without batch norm, residual scaling, sub-pixel upsampler. */
        EDSR,
        /** Blocks with SPAN's parameter-free symmetric attention, sub-pixel upsampler. */
        SPAN
    }

    private final Kind kind;
    private final int factor;
    private final int features;
    private final int blocks;
    private final float residualScale;
    private final boolean globalResidual;
    private final List<RestoreOps.Conv> head = new ArrayList<>();
    private final List<RestoreOps.Conv[]> body = new ArrayList<>();
    private final List<RestoreOps.Conv> tail = new ArrayList<>();
    private final PixelShuffle shuffle;

    /**
     * Creates a network.
     *
     * @param kind the architecture
     * @param factor the scale factor (2 or 4)
     * @param features the channel width (SRCNN: of its first layer; the second has half)
     * @param blocks the residual or attention blocks (EDSR and SPAN; ignored otherwise)
     * @param globalResidual whether to add the bicubic upsampling of the input
     * @param random the source of initial weights
     */
    public SuperResolution(Kind kind, int factor, int features, int blocks, boolean globalResidual,
            RandomGenerator random) {
        if (factor < 2) {
            throw new IllegalArgumentException("factor must be at least 2, got " + factor);
        }
        this.kind = kind;
        this.factor = factor;
        this.features = features;
        this.blocks = blocks;
        this.residualScale = 0.1f;
        this.globalResidual = globalResidual;
        this.shuffle = new PixelShuffle(factor, false);
        int r2 = factor * factor;
        float small = 1e-3f;
        switch (kind) {
            case SRCNN -> {
                head.add(new RestoreOps.Conv(1, features, 9, random));
                head.add(new RestoreOps.Conv(features, Math.max(1, features / 2), 1, random));
                tail.add(new RestoreOps.Conv(Math.max(1, features / 2), 1, 5, small, random));
            }
            case ESPCN -> {
                head.add(new RestoreOps.Conv(1, features, 5, random));
                head.add(new RestoreOps.Conv(features, Math.max(1, features / 2), 3, random));
                tail.add(new RestoreOps.Conv(Math.max(1, features / 2), r2, 3, small, random));
            }
            case EDSR, SPAN -> {
                head.add(new RestoreOps.Conv(1, features, 3, random));
                for (int b = 0; b < blocks; b++) {
                    if (kind == Kind.EDSR) {
                        body.add(new RestoreOps.Conv[] {new RestoreOps.Conv(features, features, 3, random),
                            new RestoreOps.Conv(features, features, 3, random)});
                    } else {
                        body.add(new RestoreOps.Conv[] {new RestoreOps.Conv(features, features, 3, random),
                            new RestoreOps.Conv(features, features, 3, random),
                            new RestoreOps.Conv(features, features, 3, random)});
                    }
                }
                tail.add(new RestoreOps.Conv(features, features, 3, random));
                tail.add(new RestoreOps.Conv(features, r2, 3, small, random));
            }
            default -> throw new IllegalStateException();
        }
    }

    /**
     * Describes a network for model builders on {@code [1, h, w]} inputs.
     *
     * @param kind the architecture
     * @param factor the scale factor
     * @param features the channel width
     * @param blocks the blocks (EDSR and SPAN)
     * @return the spec
     */
    public static LayerSpec spec(Kind kind, int factor, int features, int blocks) {
        return RestoreOps.spec(s -> Shape.of(1, s.dim(1) * factor, s.dim(2) * factor),
                (s, r) -> new SuperResolution(kind, factor, features, blocks, true, r));
    }

    /** @return the architecture */
    public Kind kind() {
        return kind;
    }

    /** @return the scale factor */
    public int factor() {
        return factor;
    }

    @Override
    public Var forward(Var input) {
        int h = input.shape().dim(2);
        int w = input.shape().dim(3);
        Var bicubic = Functional.resize(input, new int[] {h * factor, w * factor}, Functional.Interpolation.CUBIC,
                false);
        Var out;
        switch (kind) {
            case SRCNN -> {
                Var x = head.get(0).forward(bicubic).relu();
                x = head.get(1).forward(x).relu();
                out = tail.get(0).forward(x);
            }
            case ESPCN -> {
                Var x = head.get(0).forward(input).tanh();
                x = head.get(1).forward(x).tanh();
                out = shuffle.forward(tail.get(0).forward(x));
            }
            default -> {
                Var f = head.get(0).forward(input);
                Var x = f;
                for (RestoreOps.Conv[] block : body) {
                    if (kind == Kind.EDSR) {
                        x = x.add(block[1].forward(block[0].forward(x).relu()).mul(residualScale));
                    } else {
                        Var h1 = silu(block[0].forward(x));
                        Var h2 = silu(block[1].forward(h1));
                        Var h3 = block[2].forward(h2);
                        x = h3.add(x).mul(h3.sigmoid().sub(0.5f));
                    }
                }
                x = tail.get(0).forward(x).add(f);
                out = shuffle.forward(tail.get(1).forward(x));
            }
        }
        return globalResidual ? out.add(bicubic) : out;
    }

    private static Var silu(Var x) {
        return x.mul(x.sigmoid());
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), input.dim(1) * factor, input.dim(2) * factor);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (RestoreOps.Conv c : head) {
            p.addAll(c.parameters());
        }
        for (RestoreOps.Conv[] block : body) {
            for (RestoreOps.Conv c : block) {
                p.addAll(c.parameters());
            }
        }
        for (RestoreOps.Conv c : tail) {
            p.addAll(c.parameters());
        }
        return p;
    }

    /** @return the number of trainable scalars */
    public long parameterCount() {
        return RestoreOps.count(parameters());
    }

    /**
     * Multiply-adds of the convolutions for one low-resolution input (the bicubic skip and
     * activations not counted).
     *
     * @param h the low-resolution height
     * @param w the low-resolution width
     * @return multiply-adds
     */
    public long macs(int h, int w) {
        long low = (long) h * w;
        long high = low * factor * factor;
        long total = 0;
        if (kind == Kind.SRCNN) {
            for (RestoreOps.Conv c : head) {
                total += c.macsPerPixel() * high;
            }
            return total + tail.get(0).macsPerPixel() * high;
        }
        for (RestoreOps.Conv c : head) {
            total += c.macsPerPixel() * low;
        }
        for (RestoreOps.Conv[] block : body) {
            for (RestoreOps.Conv c : block) {
                total += c.macsPerPixel() * low;
            }
        }
        for (RestoreOps.Conv c : tail) {
            total += c.macsPerPixel() * low;
        }
        return total;
    }

    @Override
    public String toString() {
        return kind + "(x" + factor + ", " + features + " features" + (blocks > 0 && (kind == Kind.EDSR
                || kind == Kind.SPAN) ? ", " + blocks + " blocks" : "") + ")";
    }
}
