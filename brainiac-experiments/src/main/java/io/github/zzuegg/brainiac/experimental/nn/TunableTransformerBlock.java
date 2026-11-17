package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Recompute;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;
import java.util.random.RandomGenerator;

/**
 * A transformer block over {@code [batch, steps, features]} in which every normalization and
 * residual trick of the last years is a switch, so they can be compared on equal footing
 * (and used as a drop-in for {@code TransformerBlock.of(heads)}):
 *
 * <pre>{@code
 * TunableTransformerBlock.of(4)
 *     .norm(Norm.DYT)                     // LAYER, RMS, DYT (Dynamic Tanh) or NONE
 *     .placement(Placement.PERI)          // PRE, POST or PERI (sandwich: x + norm(f(norm(x))))
 *     .residual(Residual.LAYER_SCALE)     // PLAIN, REZERO, LAYER_SCALE or DEEP_NORM
 *     .position(Position.ROTARY).qkNorm(true).causal(true)
 *     .ffn(GluFeedForward.Kind.SWIGLU)
 *     .dropPath(0.1f)                     // stochastic depth
 *     .checkpoint(true);                  // recompute activations in the backward pass
 * }</pre>
 *
 * <p>Placements: pre-norm {@code x + f(norm(x))} (GPT-2 onward) trains stably; post-norm
 * {@code norm(x + f(x))} (original transformer) can generalize better but needs warm-up;
 * peri-norm, also called sandwich norm, normalizes both the sublayer's input and its output
 * before the residual add, bounding the residual stream's growth (Gemma 2/3, OLMo 2). Kim et
 * al., "Peri-LN", <a href="https://arxiv.org/abs/2502.02732">arXiv:2502.02732</a>; Ding et
 * al., CogView, <a href="https://arxiv.org/abs/2105.13290">arXiv:2105.13290</a>.
 *
 * <p>DeepNorm (Wang et al., "DeepNet: Scaling Transformers to 1,000 Layers",
 * <a href="https://arxiv.org/abs/2203.00555">arXiv:2203.00555</a>) is post-norm with the
 * residual scaled up, {@code norm(alpha x + f(x))}, and the feed-forward, value and output
 * weights scaled down by {@code beta} at initialization; see {@link DeepNorm}. Choosing it
 * forces post-norm placement.
 */
public final class TunableTransformerBlock implements Layer {

    /** The normalization layer. */
    public enum Norm {
        /** LayerNorm. */
        LAYER,
        /** RMSNorm. */
        RMS,
        /** Dynamic Tanh: no statistics, {@code gamma tanh(alpha x) + beta}. */
        DYT,
        /** No normalization at all (pair with ReZero). */
        NONE
    }

    /** Where the normalization goes relative to each sublayer. */
    public enum Placement {
        /** {@code x + f(norm(x))}. */
        PRE,
        /** {@code norm(x + f(x))}. */
        POST,
        /** Peri-LN / sandwich: {@code x + norm(f(norm(x)))}. */
        PERI
    }

    /** How the sublayer output joins the residual stream. */
    public enum Residual {
        /** {@code x + f}. */
        PLAIN,
        /** {@code x + a f} with one scalar {@code a = 0} at the start. */
        REZERO,
        /** {@code x + diag(s) f} with per-channel {@code s = 1e-2} at the start. */
        LAYER_SCALE,
        /** {@code norm(alpha x + f)} with DeepNorm's alpha and beta-scaled initialization. */
        DEEP_NORM
    }

    private final Spec spec;
    private final int features;
    private final PositionalAttention attention;
    private final GluFeedForward ffn;
    private final List<Layer> norms = new ArrayList<>();
    private final Layer attnIn;
    private final Layer attnOut;
    private final Layer ffnIn;
    private final Layer ffnOut;
    private final LayerScale attnScale;
    private final LayerScale ffnScale;
    private final float alpha;
    private final RandomGenerator random;

    private TunableTransformerBlock(Spec spec, int features, RandomGenerator random) {
        this.spec = spec;
        this.features = features;
        this.random = random;
        PositionalAttention.Settings as = PositionalAttention.Settings.of(features, spec.heads, spec.causal,
                spec.position, spec.qkNorm);
        if (spec.gridHeight > 0) {
            as = as.grid(spec.gridHeight, spec.gridWidth, spec.prefix);
        }
        attention = new PositionalAttention(as, random);
        ffn = new GluFeedForward(spec.ffn, features, Math.max(1, Math.round(features * spec.ffnFactor)), random);
        Placement placement = spec.residual == Residual.DEEP_NORM ? Placement.POST : spec.placement;
        attnIn = placement != Placement.POST ? norm() : null;
        ffnIn = placement != Placement.POST ? norm() : null;
        attnOut = placement != Placement.PRE ? norm() : null;
        ffnOut = placement != Placement.PRE ? norm() : null;
        attnScale = scale();
        ffnScale = scale();
        if (spec.residual == Residual.DEEP_NORM) {
            alpha = DeepNorm.alpha(spec.depth);
            float beta = DeepNorm.beta(spec.depth);
            for (Var w : List.of(attention.valueWeight(), attention.outputWeight())) {
                DeepNorm.scale(w, beta);
            }
            for (Var w : ffn.parameters()) {
                DeepNorm.scale(w, beta);
            }
        } else {
            alpha = 1f;
        }
    }

    private Layer norm() {
        Layer n = switch (spec.norm) {
            case LAYER -> LayerNorm.of(features);
            case RMS -> LayerNorm.rms(features);
            case DYT -> DynamicTanh.of(features);
            case NONE -> null;
        };
        if (n != null) {
            norms.add(n);
        }
        return n;
    }

    private LayerScale scale() {
        return switch (spec.residual) {
            case REZERO -> LayerScale.rezero();
            case LAYER_SCALE -> LayerScale.perChannel(features, 1e-2f);
            default -> null;
        };
    }

    /**
     * Describes a block with pre-LayerNorm, plain residuals, no positions and a GELU MLP.
     *
     * @param heads the attention heads
     * @return the spec
     */
    public static Spec of(int heads) {
        return new Spec(heads);
    }

    /** An immutable description of a block; every setter returns a new spec. */
    public static final class Spec implements LayerSpec {

        private final int heads;
        private Norm norm = Norm.LAYER;
        private Placement placement = Placement.PRE;
        private Residual residual = Residual.PLAIN;
        private PositionalAttention.Position position = PositionalAttention.Position.NONE;
        private boolean qkNorm;
        private boolean causal;
        private GluFeedForward.Kind ffn = GluFeedForward.Kind.GELU_MLP;
        private float ffnFactor = 4f;
        private float dropPath;
        private boolean checkpoint;
        private int depth = 1;
        private int gridHeight;
        private int gridWidth;
        private int prefix;

        private Spec(int heads) {
            this.heads = heads;
        }

        private Spec copy() {
            Spec s = new Spec(heads);
            s.norm = norm;
            s.placement = placement;
            s.residual = residual;
            s.position = position;
            s.qkNorm = qkNorm;
            s.causal = causal;
            s.ffn = ffn;
            s.ffnFactor = ffnFactor;
            s.dropPath = dropPath;
            s.checkpoint = checkpoint;
            s.depth = depth;
            s.gridHeight = gridHeight;
            s.gridWidth = gridWidth;
            s.prefix = prefix;
            return s;
        }

        /**
         * @param n the normalization
         * @return a new spec
         */
        public Spec norm(Norm n) {
            Spec s = copy();
            s.norm = n;
            return s;
        }

        /**
         * @param p where the normalization goes
         * @return a new spec
         */
        public Spec placement(Placement p) {
            Spec s = copy();
            s.placement = p;
            return s;
        }

        /**
         * @param r how sublayers join the residual stream
         * @return a new spec
         */
        public Spec residual(Residual r) {
            Spec s = copy();
            s.residual = r;
            return s;
        }

        /**
         * @param p the attention position encoding
         * @return a new spec
         */
        public Spec position(PositionalAttention.Position p) {
            Spec s = copy();
            s.position = p;
            return s;
        }

        /**
         * @param on whether to RMS-normalize queries and keys
         * @return a new spec
         */
        public Spec qkNorm(boolean on) {
            Spec s = copy();
            s.qkNorm = on;
            return s;
        }

        /**
         * @param on whether attention is causal
         * @return a new spec
         */
        public Spec causal(boolean on) {
            Spec s = copy();
            s.causal = on;
            return s;
        }

        /**
         * @param kind the feed-forward variant
         * @return a new spec
         */
        public Spec ffn(GluFeedForward.Kind kind) {
            Spec s = copy();
            s.ffn = kind;
            return s;
        }

        /**
         * @param factor the feed-forward hidden width as a multiple of the model width
         * @return a new spec
         */
        public Spec ffnFactor(float factor) {
            Spec s = copy();
            s.ffnFactor = factor;
            return s;
        }

        /**
         * @param rate the stochastic-depth drop rate of each branch while training
         * @return a new spec
         */
        public Spec dropPath(float rate) {
            Spec s = copy();
            s.dropPath = rate;
            return s;
        }

        /**
         * @param on whether to recompute the block's activations in the backward pass
         *     (activation checkpointing) instead of keeping them
         * @return a new spec
         */
        public Spec checkpoint(boolean on) {
            Spec s = copy();
            s.checkpoint = on;
            return s;
        }

        /**
         * @param layers the total number of blocks in the stack, for DeepNorm's constants
         * @return a new spec
         */
        public Spec depth(int layers) {
            Spec s = copy();
            s.depth = layers;
            return s;
        }

        /**
         * Sets the patch grid for {@link PositionalAttention.Position#ROTARY_2D}.
         *
         * @param height patch rows
         * @param width patch columns
         * @param prefixTokens leading non-patch tokens
         * @return a new spec
         */
        public Spec grid(int height, int width, int prefixTokens) {
            Spec s = copy();
            s.gridHeight = height;
            s.gridWidth = width;
            s.prefix = prefixTokens;
            return s;
        }

        @Override
        public Shape outputShape(Shape input) {
            if (input.rank() != 2) {
                throw new IllegalArgumentException("a transformer block expects [steps, features], got " + input);
            }
            return input;
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            return build(input.dim(input.rank() - 1), random);
        }

        /**
         * Creates the block for a known width.
         *
         * @param features the model width
         * @param random the random source
         * @return the block
         */
        public TunableTransformerBlock build(int features, RandomGenerator random) {
            return new TunableTransformerBlock(this, features, random);
        }
    }

    private static Var apply(Layer layer, Var x) {
        return layer == null ? x : layer.forward(x);
    }

    private Var sublayer(Var x, Layer in, Layer out, LayerScale scale, UnaryOperator<Var> f, boolean training) {
        if (in == null && out != null) {
            // post-norm (and DeepNorm): norm(alpha x + f(x))
            Var h = DropPath.apply(f.apply(x), spec.dropPath, training, random);
            return out.forward(x.mul(alpha).add(h));
        }
        Var h = apply(out, f.apply(apply(in, x)));
        if (scale != null) {
            h = scale.forward(h);
        }
        return x.add(DropPath.apply(h, spec.dropPath, training, random));
    }

    private Var block(Var x, boolean training) {
        x = sublayer(x, attnIn, attnOut, attnScale, attention::forward, training);
        return sublayer(x, ffnIn, ffnOut, ffnScale, ffn::forward, training);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (spec.checkpoint && training && spec.dropPath == 0) {
            return Recompute.apply(x -> block(x, true), input, parameters());
        }
        return block(input, training);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(attention.parameters());
        p.addAll(ffn.parameters());
        for (Layer n : norms) {
            p.addAll(n.parameters());
        }
        if (attnScale != null) {
            p.addAll(attnScale.parameters());
            p.addAll(ffnScale.parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "TunableTransformerBlock(" + features + ", " + spec.heads + " heads, " + spec.norm + ", "
                + spec.placement + ", " + spec.residual + ", " + spec.position + (spec.qkNorm ? ", qk-norm" : "")
                + ", " + spec.ffn + ")";
    }
}
