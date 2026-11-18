package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A U-Net over images {@code [batch, in, h, w] -> [batch, out, h, w]}: an encoder of
 * {@code levels} stages (two 3x3 convolutions with SiLU, then 2x2 max pooling, doubling the
 * channels each stage), a bottleneck, and a mirrored decoder that upsamples (nearest, then a
 * 3x3 convolution), <em>concatenates the encoder features of the same scale</em> (the skip
 * connections) and convolves again; a final 1x1 convolution maps to the outputs. The skips
 * carry precise positions past the coarse bottleneck, which is what dense prediction
 * (segmentation, denoising) needs. With {@code skips = false} it is a plain encoder-decoder
 * of the same width, the ablation baseline.
 *
 * <p>Optionally conditioned on a vector per example (a diffusion time embedding, a class):
 * {@link #forward(Var, Var)} adds a learned per-channel bias from it inside every stage, the
 * standard way diffusion U-Nets take the noise level.
 *
 * <p>Height and width must be divisible by {@code 2^levels}.
 *
 * <p>Paper: Ronneberger, Fischer and Brox, "U-Net: Convolutional Networks for Biomedical Image
 * Segmentation" (MICCAI 2015), <a href="https://arxiv.org/abs/1505.04597">arXiv:1505.04597</a>.
 */
public final class UNet2d implements Layer {

    private final int inputs;
    private final int outputs;
    private final int levels;
    private final boolean skips;
    private final int conditionFeatures;
    private final List<Stage> down = new ArrayList<>();
    private final Stage middle;
    private final List<Var[]> upConvs = new ArrayList<>();
    private final List<Stage> up = new ArrayList<>();
    private final Var head;
    private final Var headBias;

    /** Two 3x3 convolutions with SiLU and an optional condition bias after the first. */
    private static final class Stage {
        final Var w1;
        final Var b1;
        final Var w2;
        final Var b2;
        final Var cond;

        Stage(int in, int out, int conditionFeatures, RandomGenerator random) {
            w1 = ClassicOps.convWeight(random, out, in, 3);
            b1 = ClassicOps.zeros(out);
            w2 = ClassicOps.convWeight(random, out, out, 3);
            b2 = ClassicOps.zeros(out);
            cond = conditionFeatures > 0 ? ClassicOps.matrix(random, conditionFeatures, out) : null;
        }

        Var forward(Var x, Var condition) {
            Var h = ClassicOps.conv(x, w1, b1, 1);
            if (cond != null && condition != null) {
                int b = h.shape().dim(0);
                h = h.add(condition.matmul(cond).reshape(b, h.shape().dim(1), 1, 1));
            }
            h = h.mul(h.sigmoid());
            h = ClassicOps.conv(h, w2, b2, 1);
            return h.mul(h.sigmoid());
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(w1, b1, w2, b2));
            if (cond != null) {
                p.add(cond);
            }
            return p;
        }
    }

    /**
     * Creates the network.
     *
     * @param inputs the input channels
     * @param outputs the output channels
     * @param base the channels of the first stage (doubled at each level)
     * @param levels the number of downsampling stages
     * @param skips whether the decoder gets the encoder's same-scale features (a U-Net) or not
     * @param conditionFeatures the size of the conditioning vector, 0 for none
     * @param random the source of initial weights
     */
    public UNet2d(int inputs, int outputs, int base, int levels, boolean skips, int conditionFeatures,
            RandomGenerator random) {
        if (levels < 1) {
            throw new IllegalArgumentException("need at least one level, got " + levels);
        }
        this.inputs = inputs;
        this.outputs = outputs;
        this.levels = levels;
        this.skips = skips;
        this.conditionFeatures = conditionFeatures;
        int c = inputs;
        for (int l = 0; l < levels; l++) {
            int f = base << l;
            down.add(new Stage(c, f, conditionFeatures, random));
            c = f;
        }
        middle = new Stage(c, base << levels, conditionFeatures, random);
        c = base << levels;
        for (int l = levels - 1; l >= 0; l--) {
            int f = base << l;
            upConvs.add(new Var[] {ClassicOps.convWeight(random, f, c, 3), ClassicOps.zeros(f)});
            up.add(new Stage(skips ? 2 * f : f, f, conditionFeatures, random));
            c = f;
        }
        head = ClassicOps.convWeight(random, outputs, base, 1);
        headBias = ClassicOps.zeros(outputs);
    }

    /**
     * Describes an unconditioned network for model builders on {@code [inputs, h, w]}.
     *
     * @param outputs the output channels
     * @param base the first stage's channels
     * @param levels the downsampling stages
     * @param skips whether to use skip connections
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int base, int levels, boolean skips) {
        return ClassicOps.spec(s -> Shape.of(outputs, s.dim(1), s.dim(2)),
                (s, r) -> new UNet2d(s.dim(0), outputs, base, levels, skips, 0, r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, null);
    }

    /**
     * Runs the network with a conditioning vector.
     *
     * @param input {@code [batch, inputs, h, w]}
     * @param condition {@code [batch, conditionFeatures]}, or {@code null}
     * @return {@code [batch, outputs, h, w]}
     */
    public Var forward(Var input, Var condition) {
        ClassicOps.requireImages(input, inputs, this);
        int h = input.shape().dim(2);
        int w = input.shape().dim(3);
        if (h % (1 << levels) != 0 || w % (1 << levels) != 0) {
            throw new IllegalArgumentException(this + " needs sides divisible by " + (1 << levels) + ", got "
                    + input.shape());
        }
        List<Var> saved = new ArrayList<>();
        Var x = input;
        for (Stage s : down) {
            x = s.forward(x, condition);
            saved.add(x);
            x = Functional.pool(x, Functional.PoolKind.MAX, new int[] {2, 2}, new int[] {2, 2}, new int[] {0, 0},
                    new int[] {0, 0}, new int[] {1, 1}, false, false, 1f);
        }
        x = middle.forward(x, condition);
        for (int i = 0; i < levels; i++) {
            Var skip = saved.get(levels - 1 - i);
            x = Functional.resize(x, new int[] {skip.shape().dim(2), skip.shape().dim(3)},
                    Functional.Interpolation.NEAREST, false);
            Var[] uc = upConvs.get(i);
            x = ClassicOps.conv(x, uc[0], uc[1], 1);
            if (skips) {
                x = Var.concat(1, skip, x);
            }
            x = up.get(i).forward(x, condition);
        }
        return ClassicOps.conv(x, head, headBias, 1);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(outputs, input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Stage s : down) {
            p.addAll(s.parameters());
        }
        p.addAll(middle.parameters());
        for (int i = 0; i < levels; i++) {
            p.addAll(List.of(upConvs.get(i)));
            p.addAll(up.get(i).parameters());
        }
        p.add(head);
        p.add(headBias);
        return p;
    }

    /** @return the size of the conditioning vector, 0 for none */
    public int conditionFeatures() {
        return conditionFeatures;
    }

    /** @return whether the decoder uses skip connections */
    public boolean skips() {
        return skips;
    }

    @Override
    public String toString() {
        return "UNet2d(" + inputs + " -> " + outputs + ", " + levels + " levels" + (skips ? "" : ", no skips") + ")";
    }
}
