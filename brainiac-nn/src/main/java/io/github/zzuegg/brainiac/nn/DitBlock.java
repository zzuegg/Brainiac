package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A Diffusion Transformer block with adaLN-Zero conditioning (DiT, Peebles and Xie, 2023), as
 * F5-TTS and other flow-matching and diffusion models stack them over sequences. A condition
 * vector (such as a timestep embedding) is mapped, through SiLU and one linear layer, to a
 * shift, a scale and a gate for each of the two sub-layers:
 * <pre>
 * x += gate_a * Attention(LayerNorm(x) * (1 + scale_a) + shift_a)
 * x += gate_f * MLP(LayerNorm(x) * (1 + scale_f) + shift_f)
 * </pre>
 * The layer norms have no weights of their own; attention is bidirectional, with optional
 * rotary positions on queries and keys. With random weights the conditioning map starts at
 * zero, so the block starts as the identity.
 *
 * <p>Inputs are a sequence {@code [length, features]} and a condition {@code [condition]};
 * the output has the sequence's shape. Wire it with {@code Node.merge(DitBlock.of(heads), cond)}.
 */
public final class DitBlock implements MultiInputLayer {

    /** How positions enter the attention. */
    public enum Rotary {
        /** no rotary embedding */
        NONE,
        /** feature {@code i} turns with feature {@code i + d / 2}, as {@link Functional#rotary} */
        HALF,
        /** neighbouring features turn together, as {@link Functional#rotaryInterleaved} (F5-TTS) */
        INTERLEAVED
    }

    /**
     * The block's shape.
     *
     * @param heads the attention heads
     * @param rotary how positions enter the attention
     * @param rotaryBase the rotary frequency base, such as 10000
     * @param activation the MLP's activation (F5-TTS: GELU with the tanh approximation)
     * @param eps the layer norms' epsilon
     */
    public record Settings(int heads, Rotary rotary, float rotaryBase, Activation activation, float eps) {
        /** Checks the values. */
        public Settings {
            Objects.requireNonNull(rotary, "rotary");
            Objects.requireNonNull(activation, "activation");
            if (heads < 1) {
                throw new IllegalArgumentException("a DiT block needs at least one head, got " + heads);
            }
        }
    }

    private final Settings settings;
    private final Var modWeight;
    private final Var modBias;
    private final Var wq;
    private final Var bq;
    private final Var wk;
    private final Var bk;
    private final Var wv;
    private final Var bv;
    private final Var wo;
    private final Var bo;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    /**
     * A block from given weights; linear weights are {@code [in, out]}.
     *
     * @param settings the shape
     * @param modWeight the conditioning map {@code [condition, 6 features]}: shift, scale and
     *     gate of the attention, then of the MLP
     * @param modBias its bias {@code [6 features]}
     * @param wq the query projection {@code [features, inner]}
     * @param bq its bias
     * @param wk the key projection {@code [features, inner]}
     * @param bk its bias
     * @param wv the value projection {@code [features, inner]}
     * @param bv its bias
     * @param wo the output projection {@code [inner, features]}
     * @param bo its bias
     * @param w1 the MLP's first layer {@code [features, hidden]}
     * @param b1 its bias
     * @param w2 the MLP's second layer {@code [hidden, features]}
     * @param b2 its bias
     */
    public DitBlock(Settings settings, Var modWeight, Var modBias, Var wq, Var bq, Var wk, Var bk, Var wv, Var bv,
            Var wo, Var bo, Var w1, Var b1, Var w2, Var b2) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int d = wq.shape().dim(0);
        int inner = wq.shape().dim(1);
        if (modWeight.shape().dim(1) != 6 * d || inner % settings.heads() != 0) {
            throw new IllegalArgumentException("a DiT block of " + d + " features needs a [condition, " + 6 * d
                    + "] conditioning map and heads dividing " + inner + ", got " + modWeight.shape());
        }
        if (settings.rotary() != Rotary.NONE && (inner / settings.heads()) % 2 != 0) {
            throw new IllegalArgumentException("rotary embeddings need an even head size, got " + inner / settings.heads());
        }
        this.modWeight = modWeight;
        this.modBias = modBias;
        this.wq = wq;
        this.bq = bq;
        this.wk = wk;
        this.bk = bk;
        this.wv = wv;
        this.bv = bv;
        this.wo = wo;
        this.bo = bo;
        this.w1 = w1;
        this.b1 = b1;
        this.w2 = w2;
        this.b2 = b2;
    }

    /**
     * A block with random weights; the conditioning map starts at zero.
     *
     * @param settings the shape
     * @param features the sequence's features
     * @param condition the condition's width
     * @param headSize the size of each head
     * @param hidden the MLP's width
     * @param random the random source
     */
    public DitBlock(Settings settings, int features, int condition, int headSize, int hidden, SplittableRandom random) {
        this(settings, Var.param(Tensor.zeros(Shape.of(condition, 6 * features))), zeros(6 * features),
                dense(features, settings.heads() * headSize, random), zeros(settings.heads() * headSize),
                dense(features, settings.heads() * headSize, random), zeros(settings.heads() * headSize),
                dense(features, settings.heads() * headSize, random), zeros(settings.heads() * headSize),
                dense(settings.heads() * headSize, features, random), zeros(features),
                dense(features, hidden, random), zeros(hidden), dense(hidden, features, random), zeros(features));
    }

    private static Var dense(int in, int out, SplittableRandom random) {
        return Var.param(Init.XAVIER.create(Shape.of(in, out), in, out, random));
    }

    private static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    /**
     * Describes a block with interleaved rotary positions (base 10000), a tanh-GELU MLP twice
     * the width and heads of {@code features / heads}.
     *
     * @param heads the attention heads
     * @return the spec
     */
    public static MultiInputLayer.Spec of(int heads) {
        return of(new Settings(heads, Rotary.INTERLEAVED, 10_000f, Activation.GELU_TANH, 1e-6f), 2);
    }

    /**
     * Describes a block.
     *
     * @param settings the shape
     * @param mlpRatio the MLP's width over the features
     * @return the spec
     */
    public static MultiInputLayer.Spec of(Settings settings, int mlpRatio) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return check(inputs, -1);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                check(inputs, -1);
                int d = inputs.get(0).dim(1);
                if (d % settings.heads() != 0) {
                    throw new IllegalArgumentException(settings.heads() + " heads do not divide " + d + " features");
                }
                return new DitBlock(settings, d, (int) inputs.get(1).size(), d / settings.heads(), mlpRatio * d, random);
            }
        };
    }

    private static Shape check(List<Shape> inputs, int condition) {
        if (inputs.size() != 2 || inputs.get(0).rank() != 2
                || (condition >= 0 && inputs.get(1).size() != condition)) {
            throw new IllegalArgumentException("a DiT block takes a sequence [length, features] and a condition"
                    + (condition >= 0 ? " [" + condition + "]" : "") + ", got " + inputs);
        }
        return inputs.get(0);
    }

    /** @return the shape */
    public Settings settings() {
        return settings;
    }

    /** @return the sequence's features */
    public int features() {
        return wq.shape().dim(0);
    }

    /**
     * Runs the block.
     *
     * @param x {@code [batch, length, features]}
     * @param condition {@code [batch, condition]}
     * @return {@code [batch, length, features]}
     */
    public Var forward(Var x, Var condition) {
        int n = x.shape().dim(0);
        int len = x.shape().dim(1);
        int d = features();
        int inner = wq.shape().dim(1);
        int heads = settings.heads();
        int hs = inner / heads;
        Var e = Activation.SILU.forward(condition.reshape(n, modWeight.shape().dim(0))).matmul(modWeight).add(modBias);
        Var norm = Functional.adaLayerNorm(x, e.slice(1, d, 2 * d), e.slice(1, 0, d), settings.eps());
        Var q = heads(norm.matmul(wq).add(bq), n, len, heads, hs);
        Var k = heads(norm.matmul(wk).add(bk), n, len, heads, hs);
        Var v = heads(norm.matmul(wv).add(bv), n, len, heads, hs);
        if (settings.rotary() == Rotary.HALF) {
            q = Functional.rotary(q, settings.rotaryBase(), 0);
            k = Functional.rotary(k, settings.rotaryBase(), 0);
        } else if (settings.rotary() == Rotary.INTERLEAVED) {
            q = Functional.rotaryInterleaved(q, settings.rotaryBase(), 0);
            k = Functional.rotaryInterleaved(k, settings.rotaryBase(), 0);
        }
        Var a = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(hs)))
                .permute(0, 2, 1, 3).reshape(n, len, inner).matmul(wo).add(bo);
        x = Functional.gate(x, a, e.slice(1, 2 * d, 3 * d));
        norm = Functional.adaLayerNorm(x, e.slice(1, 4 * d, 5 * d), e.slice(1, 3 * d, 4 * d), settings.eps());
        Var f = settings.activation().forward(norm.matmul(w1).add(b1)).matmul(w2).add(b2);
        return Functional.gate(x, f, e.slice(1, 5 * d, 6 * d));
    }

    private static Var heads(Var x, int n, int len, int heads, int size) {
        return x.reshape(n, len, heads, size).permute(0, 2, 1, 3);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException(this + " takes a sequence and a condition, got " + inputs.size()
                    + " inputs");
        }
        return forward(inputs.get(0), inputs.get(1));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return check(inputs, modWeight.shape().dim(0));
    }

    @Override
    public Shape outputShape(Shape input) {
        throw new IllegalStateException("a DiT block needs a condition; wire it with Node.merge(spec, condition)");
    }

    @Override
    public List<Var> parameters() {
        return List.of(modWeight, modBias, wq, bq, wk, bk, wv, bv, wo, bo, w1, b1, w2, b2);
    }

    @Override
    public String toString() {
        return "DitBlock[" + features() + " features, " + settings.heads() + " heads, condition "
                + modWeight.shape().dim(0) + ", rotary " + settings.rotary().name().toLowerCase(Locale.ROOT) + "]";
    }

    static Config write(DitBlock b) {
        Settings s = b.settings;
        return Config.builder().put("features", b.features()).put("condition", b.modWeight.shape().dim(0))
                .put("inner", b.wq.shape().dim(1)).put("hidden", b.w1.shape().dim(1)).put("heads", s.heads())
                .put("rotary", s.rotary().name()).put("rotaryBase", s.rotaryBase())
                .put("activation", s.activation()).put("eps", s.eps()).build();
    }

    static DitBlock read(Config c) {
        Settings s = new Settings(c.getInt("heads"), Rotary.valueOf(c.getString("rotary")), c.getFloat("rotaryBase"),
                c.getActivation("activation"), c.getFloat("eps"));
        return new DitBlock(s, c.getInt("features"), c.getInt("condition"), c.getInt("inner") / s.heads(),
                c.getInt("hidden"), new SplittableRandom());
    }
}
