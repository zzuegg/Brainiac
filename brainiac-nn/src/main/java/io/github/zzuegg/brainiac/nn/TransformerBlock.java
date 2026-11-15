package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A transformer layer over sequences {@code [steps, features]}: self-attention and a
 * feed-forward network, each with a residual connection and normalization. As a decoder
 * block ({@link Spec#crossAttention(boolean)}), it also attends to a second input, the
 * encoder output, wired with {@code Node.merge(spec, memory)}.
 *
 * <pre>{@code
 * TransformerBlock.of(4)                    // 4 heads
 *     .ffnSize(512)                         // default 4 x features
 *     .activation(Activation.GELU)          // feed-forward activation
 *     .gated(true)                          // GLU-style feed-forward (SwiGLU with SILU)
 *     .rmsNorm(true)                        // RMSNorm instead of LayerNorm
 *     .preNorm(true)                        // normalize before each sublayer (default) or after
 *     .causal(true).rotary(10000)           // decoder-only (GPT-style) block with RoPE
 *     .kvHeads(1).alibi(false)
 *     .dropout(0.1f);
 * }</pre>
 */
public final class TransformerBlock implements MultiInputLayer {

    /**
     * Every setting, as stored in model files.
     *
     * @param features the features per step
     * @param memoryFeatures the features of the cross-attention input, or 0 for none
     * @param heads the attention heads
     * @param kvHeads the key/value heads
     * @param ffnSize the hidden size of the feed-forward network
     * @param activation the feed-forward activation
     * @param gated whether the feed-forward network is gated
     * @param rmsNorm whether to use RMSNorm
     * @param preNorm whether to normalize before each sublayer
     * @param causal whether self-attention is causal
     * @param rotaryBase the RoPE base, or 0
     * @param alibi whether to add ALiBi biases
     * @param dropout the dropout rate
     * @param bias whether projections have biases
     */
    public record Settings(int features, int memoryFeatures, int heads, int kvHeads, int ffnSize,
            Activation activation, boolean gated, boolean rmsNorm, boolean preNorm, boolean causal, float rotaryBase,
            boolean alibi, float dropout, boolean bias) {
    }

    private final Settings settings;
    private final MultiHeadAttention selfAttention;
    private final MultiHeadAttention crossAttention;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final LayerNorm norm3;
    private final Dense up;
    private final Dense gate;
    private final Dense down;
    private final RandomGenerator random;

    /**
     * Creates a block with fresh weights.
     *
     * @param s the settings
     * @param random the random source
     */
    public TransformerBlock(Settings s, SplittableRandom random) {
        this.settings = Objects.requireNonNull(s, "settings");
        int d = s.features();
        MultiHeadAttention.Settings self = new MultiHeadAttention.Settings(d, d, s.heads(), s.kvHeads(),
                d / s.heads(), d / s.heads(), d, s.bias(), s.causal(), s.rotaryBase(), s.alibi(), s.dropout(), 0,
                false);
        if (d % s.heads() != 0) {
            throw new IllegalArgumentException(s.heads() + " heads do not divide " + d + " features");
        }
        selfAttention = new MultiHeadAttention(self, Init.XAVIER, random);
        crossAttention = s.memoryFeatures() > 0 ? new MultiHeadAttention(new MultiHeadAttention.Settings(d,
                s.memoryFeatures(), s.heads(), s.kvHeads(), d / s.heads(), d / s.heads(), d, s.bias(), false, 0,
                false, s.dropout(), 0, false), Init.XAVIER, random) : null;
        norm1 = norm(d);
        norm2 = norm(d);
        norm3 = crossAttention != null ? norm(d) : null;
        up = new Dense(d, s.ffnSize(), Activation.IDENTITY, s.activation().preferredInit(), random);
        gate = s.gated() ? new Dense(d, s.ffnSize(), Activation.IDENTITY, Init.XAVIER, random) : null;
        down = new Dense(s.ffnSize(), d, Activation.IDENTITY, Init.XAVIER, random);
        this.random = random.split();
    }

    private LayerNorm norm(int d) {
        return settings.rmsNorm() ? LayerNorm.rms(d) : LayerNorm.of(d);
    }

    /**
     * Describes a transformer block.
     *
     * @param heads the number of attention heads
     * @return the spec
     */
    public static Spec of(int heads) {
        return new Spec(heads);
    }

    /** An immutable description; every setter returns a new spec. */
    public static final class Spec implements LayerSpec, MultiInputLayer.Spec {

        private final int heads;
        private int kvHeads = -1;
        private int ffnSize = -1;
        private Activation activation = Activation.GELU;
        private boolean gated;
        private boolean rmsNorm;
        private boolean preNorm = true;
        private boolean causal;
        private float rotaryBase;
        private boolean alibi;
        private float dropout;
        private boolean bias = true;
        private boolean cross;

        private Spec(int heads) {
            if (heads < 1) {
                throw new IllegalArgumentException("heads must be at least 1, got " + heads);
            }
            this.heads = heads;
        }

        private Spec copy() {
            Spec s = new Spec(heads);
            s.kvHeads = kvHeads;
            s.ffnSize = ffnSize;
            s.activation = activation;
            s.gated = gated;
            s.rmsNorm = rmsNorm;
            s.preNorm = preNorm;
            s.causal = causal;
            s.rotaryBase = rotaryBase;
            s.alibi = alibi;
            s.dropout = dropout;
            s.bias = bias;
            s.cross = cross;
            return s;
        }

        /** @param n key/value heads @return the new spec */
        public Spec kvHeads(int n) {
            Spec s = copy();
            s.kvHeads = n;
            return s;
        }

        /** @param size the feed-forward hidden size @return the new spec */
        public Spec ffnSize(int size) {
            Spec s = copy();
            s.ffnSize = size;
            return s;
        }

        /** @param a the feed-forward activation; parameter-free activations only @return the new spec */
        public Spec activation(Activation a) {
            Spec s = copy();
            s.activation = Objects.requireNonNull(a, "activation");
            return s;
        }

        /** @param gated whether the feed-forward network is gated (GLU family) @return the new spec */
        public Spec gated(boolean gated) {
            Spec s = copy();
            s.gated = gated;
            return s;
        }

        /** @param rms whether to use RMSNorm instead of LayerNorm @return the new spec */
        public Spec rmsNorm(boolean rms) {
            Spec s = copy();
            s.rmsNorm = rms;
            return s;
        }

        /** @param pre normalize before each sublayer ({@code true}) or after @return the new spec */
        public Spec preNorm(boolean pre) {
            Spec s = copy();
            s.preNorm = pre;
            return s;
        }

        /** @param causal whether self-attention masks future steps @return the new spec */
        public Spec causal(boolean causal) {
            Spec s = copy();
            s.causal = causal;
            return s;
        }

        /** @param base the RoPE base, 0 for none @return the new spec */
        public Spec rotary(float base) {
            Spec s = copy();
            s.rotaryBase = base;
            return s;
        }

        /** @param alibi whether to add ALiBi biases @return the new spec */
        public Spec alibi(boolean alibi) {
            Spec s = copy();
            s.alibi = alibi;
            return s;
        }

        /** @param rate the dropout rate on attention weights and sublayer outputs @return the new spec */
        public Spec dropout(float rate) {
            Dropout.check(rate);
            Spec s = copy();
            s.dropout = rate;
            return s;
        }

        /** @param bias whether projections have biases @return the new spec */
        public Spec bias(boolean bias) {
            Spec s = copy();
            s.bias = bias;
            return s;
        }

        /**
         * Adds cross-attention to a second input (decoder block).
         *
         * @param cross whether to add it
         * @return the new spec
         */
        public Spec crossAttention(boolean cross) {
            Spec s = copy();
            s.cross = cross;
            return s;
        }

        Settings settings(List<Shape> inputs) {
            if (inputs.size() != (cross ? 2 : 1)) {
                throw new IllegalArgumentException(cross ? "a decoder block takes the sequence and the encoder output"
                        : "a transformer block takes one input; set crossAttention(true) for two");
            }
            Shape x = inputs.get(0);
            if (x.rank() != 2 || (cross && inputs.get(1).rank() != 2)) {
                throw new IllegalArgumentException("a transformer block expects [steps, features], got " + inputs);
            }
            int d = x.dim(1);
            return new Settings(d, cross ? inputs.get(1).dim(1) : 0, heads, kvHeads > 0 ? kvHeads : heads,
                    ffnSize > 0 ? ffnSize : 4 * d, activation, gated, rmsNorm, preNorm, causal, rotaryBase, alibi,
                    dropout, bias);
        }

        @Override
        public Shape outputShape(List<Shape> inputs) {
            settings(inputs);
            return inputs.get(0);
        }

        @Override
        public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
            return new TransformerBlock(settings(inputs), random);
        }

        @Override
        public Shape outputShape(Shape input) {
            return outputShape(List.of(input));
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            return create(List.of(input), random);
        }
    }

    private Var drop(Var x, boolean training) {
        float rate = settings.dropout();
        if (!training || rate == 0) {
            return x;
        }
        Tensor keep = Tensor.uniform(x.shape(), 0, 1, random).to(x.value().device())
                .greater(Tensor.full(Shape.of(), rate, x.value().device()));
        return x.mul(keep.div(1 - rate));
    }

    private Var ffn(Var x) {
        Var h = settings.activation().forward(up.forward(x));
        if (gate != null) {
            h = h.mul(gate.forward(x));
        }
        return down.forward(h);
    }

    private Var sublayer(Var x, LayerNorm norm, java.util.function.UnaryOperator<Var> f, boolean training) {
        if (settings.preNorm()) {
            return x.add(drop(f.apply(norm.forward(x)), training));
        }
        return norm.forward(x.add(drop(f.apply(x), training)));
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        x = sublayer(x, norm1, h -> selfAttention.forward(List.of(h), training), training);
        if (crossAttention != null) {
            Var memory = inputs.get(1);
            x = sublayer(x, norm3, h -> crossAttention.forward(List.of(h, memory), training), training);
        }
        return sublayer(x, norm2, this::ffn, training);
    }

    @Override
    public Var forward(Var input) {
        return forward(List.of(input), false);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(selfAttention.parameters());
        if (crossAttention != null) {
            p.addAll(crossAttention.parameters());
            p.addAll(norm3.parameters());
        }
        p.addAll(norm1.parameters());
        p.addAll(norm2.parameters());
        p.addAll(up.parameters());
        if (gate != null) {
            p.addAll(gate.parameters());
        }
        p.addAll(down.parameters());
        return p;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    @Override
    public String toString() {
        return "TransformerBlock(" + settings.features() + ", " + settings.heads() + " heads"
                + (settings.causal() ? ", causal" : "") + (crossAttention != null ? ", cross" : "") + ")";
    }

    static Config write(TransformerBlock b) {
        Settings s = b.settings;
        return Config.builder()
                .put("features", s.features()).put("memoryFeatures", s.memoryFeatures())
                .put("heads", s.heads()).put("kvHeads", s.kvHeads()).put("ffnSize", s.ffnSize())
                .put("activation", s.activation()).put("gated", s.gated()).put("rmsNorm", s.rmsNorm())
                .put("preNorm", s.preNorm()).put("causal", s.causal()).put("rotaryBase", s.rotaryBase())
                .put("alibi", s.alibi()).put("dropout", s.dropout()).put("bias", s.bias())
                .build();
    }

    static TransformerBlock read(Config c) {
        return new TransformerBlock(new Settings(c.getInt("features"), c.getInt("memoryFeatures"), c.getInt("heads"),
                c.getInt("kvHeads"), c.getInt("ffnSize"), c.getActivation("activation"), c.getBoolean("gated"),
                c.getBoolean("rmsNorm"), c.getBoolean("preNorm"), c.getBoolean("causal"), c.getFloat("rotaryBase"),
                c.getBoolean("alibi"), c.getFloat("dropout"), c.getBoolean("bias")), new SplittableRandom());
    }
}
