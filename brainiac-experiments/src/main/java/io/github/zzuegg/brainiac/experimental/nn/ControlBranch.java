package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * ControlNet-style conditioning (Zhang, Rao and Agrawala,
 * <a href="https://arxiv.org/abs/2302.05543">arXiv:2302.05543</a>) for a residual MLP denoiser:
 * the pretrained {@link Backbone} stays frozen; a <em>trainable copy</em> of its input layer
 * and residual blocks sees the noisy input plus the control map (through a zero-initialised
 * projection), and each copied block's output joins the frozen block's output through a
 * zero-initialised linear layer ("zero convolution"). At the start of training the model is
 * therefore exactly the base model, and the gradient still reaches every branch weight through
 * the zero layers, so the control is learned without disturbing what the base already knows.
 *
 * <p>As a {@link Denoiser.Conditional} the condition is the control map, {@code [n, control]}.
 * Only the branch's weights are in {@link #parameters()}; the base receives no gradient.
 */
public final class ControlBranch implements Denoiser.Conditional {
    private static final int TIME_HALF = 8;

    /**
     * A residual MLP denoiser {@code x, t (, condition) -> prediction}: an input layer on the
     * data, sinusoidal time features and an optional condition vector, {@code depth} residual
     * SiLU blocks, and a linear output. It is the base model of a {@link ControlBranch}, and
     * with a condition it is also the "concatenate the condition" baseline.
     */
    public static final class Backbone implements Denoiser.Conditional, Denoiser {
        final int dim;
        final int condition;
        final float timeScale;
        final Var inX;
        final Var inT;
        final Var inC;
        final Var inB;
        final List<Var> blockW = new ArrayList<>();
        final List<Var> blockB = new ArrayList<>();
        final Var outW;
        final Var outB;

        private Backbone(int dim, int width, int depth, int condition, float timeScale, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            this.dim = dim;
            this.condition = condition;
            this.timeScale = timeScale;
            inX = TechniqueOps.weight(r, dim, width);
            inT = TechniqueOps.weight(r, 2 * TIME_HALF, width);
            inC = condition > 0 ? TechniqueOps.weight(r, condition, width) : null;
            inB = TechniqueOps.filled(0f, width);
            for (int i = 0; i < depth; i++) {
                blockW.add(TechniqueOps.weight(r, width, width));
                blockB.add(TechniqueOps.filled(0f, width));
            }
            outW = TechniqueOps.weight(r, width, dim);
            outB = TechniqueOps.filled(0f, dim);
        }

        private Backbone(Backbone base, int condition) {
            this.dim = base.dim;
            this.condition = condition;
            this.timeScale = base.timeScale;
            inX = copy(base.inX);
            inT = copy(base.inT);
            inC = condition > 0 ? TechniqueOps.filled(0f, condition, base.inB.shape().dim(0)) : null;
            inB = copy(base.inB);
            for (int i = 0; i < base.blockW.size(); i++) {
                blockW.add(copy(base.blockW.get(i)));
                blockB.add(copy(base.blockB.get(i)));
            }
            outW = copy(base.outW);
            outB = copy(base.outB);
        }

        private static Var copy(Var p) {
            return Var.param(Tensor.of(p.value().shape(), p.value().toArray()));
        }

        /**
         * A fresh backbone.
         *
         * @param dim the data dimension
         * @param width the hidden width
         * @param depth the number of residual blocks
         * @param condition the condition size, 0 for none
         * @param timeScale multiplies the time before its sinusoidal features, such as
         *     {@code 1 / steps} for {@code Diffusion}
         * @param seed the initialization seed
         * @return the backbone
         */
        public static Backbone of(int dim, int width, int depth, int condition, float timeScale, long seed) {
            return new Backbone(dim, width, depth, condition, timeScale, seed);
        }

        /**
         * A trainable copy of these weights with an added, zero-initialised condition input:
         * the "fine-tune the full model" baseline, equal to this model at the start.
         *
         * @param condition the condition size
         * @return the copy
         */
        public Backbone withCondition(int condition) {
            return new Backbone(this, condition);
        }

        /** @return the trainable parameters */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(inX, inT, inB));
            if (inC != null) {
                p.add(inC);
            }
            for (int i = 0; i < blockW.size(); i++) {
                p.add(blockW.get(i));
                p.add(blockB.get(i));
            }
            p.add(outW);
            p.add(outB);
            return p;
        }

        private Tensor time(Tensor time) {
            float[] t = time.toArray();
            for (int i = 0; i < t.length; i++) {
                t[i] *= timeScale;
            }
            return GenerativeOps.timeFeatures(t, TIME_HALF, 30);
        }

        private static Var w(Var p, boolean frozen) {
            return frozen ? Var.of(p.value()) : p;
        }

        /** The input layer's activation, with an optional extra pre-activation term. */
        Var input(Var x, Tensor time, Tensor cond, Var extra, boolean frozen) {
            Var h = x.matmul(w(inX, frozen)).add(Var.of(time(time)).matmul(w(inT, frozen))).add(w(inB, frozen));
            if (inC != null && cond != null) {
                h = h.add(Var.of(cond).matmul(w(inC, frozen)));
            }
            if (extra != null) {
                h = h.add(extra);
            }
            return GenerativeOps.silu(h);
        }

        Var block(int i, Var h, boolean frozen) {
            return h.add(GenerativeOps.silu(h.matmul(w(blockW.get(i), frozen)).add(w(blockB.get(i), frozen))));
        }

        Var output(Var h, boolean frozen) {
            return h.matmul(w(outW, frozen)).add(w(outB, frozen));
        }

        @Override
        public Var apply(Var noisy, Tensor time, Tensor cond) {
            Var h = input(noisy, time, cond, null, false);
            for (int i = 0; i < blockW.size(); i++) {
                h = block(i, h, false);
            }
            return output(h, false);
        }

        @Override
        public Var apply(Var noisy, Tensor time) {
            return apply(noisy, time, null);
        }
    }

    private final Backbone base;
    private final Backbone copy;
    private final Var controlIn;
    private final List<Var> zeroW = new ArrayList<>();
    private final List<Var> zeroB = new ArrayList<>();

    private ControlBranch(Backbone base, int control) {
        this.base = base;
        this.copy = new Backbone(base, 0);
        int width = base.inB.shape().dim(0);
        controlIn = TechniqueOps.filled(0f, control, width);
        for (int i = 0; i < base.blockW.size(); i++) {
            zeroW.add(TechniqueOps.filled(0f, width, width));
            zeroB.add(TechniqueOps.filled(0f, width));
        }
    }

    /**
     * Attaches a control branch to a pretrained backbone, which is then used frozen.
     *
     * @param base the pretrained base model, without a condition input
     * @param control the control map size
     * @return the conditioned model; equal to {@code base} until trained
     */
    public static ControlBranch of(Backbone base, int control) {
        return new ControlBranch(base, control);
    }

    /** @return the branch's parameters: the copy, its control input and the zero layers */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(copy.parameters());
        p.remove(copy.outW);
        p.remove(copy.outB);
        p.add(controlIn);
        p.addAll(zeroW);
        p.addAll(zeroB);
        return p;
    }

    /** @return the frozen base model */
    public Backbone base() {
        return base;
    }

    @Override
    public Var apply(Var noisy, Tensor time, Tensor control) {
        Var c = copy.input(noisy, time, null, Var.of(control).matmul(controlIn), false);
        Var h = base.input(noisy, time, null, null, true);
        for (int i = 0; i < base.blockW.size(); i++) {
            c = copy.block(i, c, false);
            h = base.block(i, h, true).add(c.matmul(zeroW.get(i)).add(zeroB.get(i)));
        }
        return base.output(h, true);
    }
}
