package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Multi-head self-attention over {@code [batch, steps, features]} in which the recent
 * refinements of the attention weights are switches, so they can be compared in one
 * otherwise identical layer (a drop-in for {@code MultiHeadAttention.of(heads)}):
 *
 * <pre>{@code
 * RefinedAttention.of(4).causal(true)
 *     .rotary(10000)                         // RoPE, or 0 for NoPE
 *     .weighting(Weighting.SOFTPICK)         // SOFTMAX, SIGMOID, SOFTPICK or STICK_BREAKING
 *     .scalableSoftmax(true)                 // SSMax: logits times s log n
 *     .sinkLogit(true)                       // learned per-head "attend to nothing" logit
 *     .registers(2)                          // learned key/value register slots
 *     .talkingHeads(true)                    // head mixing before and after the weighting
 *     .selective(true)                       // selective attention (parameter free)
 *     .window(8);                            // sliding-window (local) attention
 * }</pre>
 *
 * <p>The weightings are {@link Softpick}, {@link SigmoidAttention} and
 * {@link StickBreakingAttention}; the other options are {@link ScalableSoftmax},
 * {@link TalkingHeadsAttention} and {@link SelectiveAttention}. A <b>sink logit</b> is a
 * learned per-head logit appended as an extra key with no value (as in gpt-oss and in
 * "off-by-one" softmax): a head can put its mass there and output nothing, instead of
 * misusing the first token as a sink. <b>Registers</b> are learned key/value slots every
 * query can see, an attention-level form of the register tokens of Darcet et al., "Vision
 * Transformers Need Registers", <a href="https://arxiv.org/abs/2309.16588">arXiv:2309.16588</a>.
 *
 * <p>For analysis, the layer keeps the attention weights of its last forward pass
 * ({@link #lastWeights}), and {@link #restrict} limits which keys each query may see (for
 * evaluating cache eviction policies).
 */
public final class RefinedAttention implements Layer {

    /** How scores become attention weights. */
    public enum Weighting {
        /** The softmax over keys. */
        SOFTMAX,
        /** Element-wise {@code sigmoid(score - log n)}. */
        SIGMOID,
        /** The rectified, not-sum-to-one {@link Softpick}. */
        SOFTPICK,
        /** {@link StickBreakingAttention}; causal only, without sink or registers. */
        STICK_BREAKING
    }

    /**
     * Every setting.
     *
     * @param features the model width
     * @param heads the number of heads, dividing {@code features}
     * @param causal whether each step sees only itself and earlier steps
     * @param rotaryBase the RoPE base, or 0 for no position encoding
     * @param weighting how scores become weights
     * @param scalableSoftmax whether to scale logits by a learned {@code s log n}
     * @param sinkLogit whether to add a learned per-head sink logit
     * @param registers the number of learned key/value register slots
     * @param talkingHeads whether to mix heads before and after the weighting
     * @param selective whether to apply selective attention
     * @param window the sliding window (keys a query sees, itself included), or 0 for all
     */
    public record Settings(int features, int heads, boolean causal, float rotaryBase, Weighting weighting,
            boolean scalableSoftmax, boolean sinkLogit, int registers, boolean talkingHeads, boolean selective,
            int window) {

        /**
         * Validates the settings.
         *
         * @throws IllegalArgumentException for impossible combinations
         */
        public Settings {
            if (heads < 1 || features % heads != 0) {
                throw new IllegalArgumentException(heads + " heads do not divide " + features + " features");
            }
            if (rotaryBase > 0 && (features / heads) % 2 != 0) {
                throw new IllegalArgumentException("RoPE needs an even head size");
            }
            if (registers < 0 || window < 0) {
                throw new IllegalArgumentException("registers and window must be at least 0");
            }
            if (weighting == Weighting.STICK_BREAKING && (!causal || sinkLogit || registers > 0)) {
                throw new IllegalArgumentException("stick-breaking attention is causal, without sink or registers");
            }
            if (selective && !causal) {
                throw new IllegalArgumentException("selective attention is causal");
            }
        }
    }

    private final Settings s;
    private final int headSize;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var ssmaxGain;
    private final Var sink;
    private final Var registerKeys;
    private final Var registerValues;
    private final Var preMix;
    private final Var postMix;
    private Tensor lastWeights;
    private Tensor lastPenalty;
    private Tensor restriction;

    /**
     * Creates the layer.
     *
     * @param settings the settings
     * @param random the random source
     */
    public RefinedAttention(Settings settings, RandomGenerator random) {
        this.s = settings;
        int d = s.features();
        int h = s.heads();
        this.headSize = d / h;
        wq = MixerOps.weight(random, d, d);
        wk = MixerOps.weight(random, d, d);
        wv = MixerOps.weight(random, d, d);
        wo = MixerOps.weight(random, d, d);
        ssmaxGain = s.scalableSoftmax() ? MixerOps.filled(ScalableSoftmax.INITIAL_GAIN, h) : null;
        sink = s.sinkLogit() ? MixerOps.filled(0f, h) : null;
        registerKeys = s.registers() > 0 ? TechniqueOps.normal(random, 1f, 1, h, s.registers(), headSize) : null;
        registerValues = s.registers() > 0 ? TechniqueOps.normal(random, 0.1f, 1, h, s.registers(), headSize) : null;
        preMix = s.talkingHeads() ? TalkingHeadsAttention.identityMixing(h, 0.05f, random) : null;
        postMix = s.talkingHeads() ? TalkingHeadsAttention.identityMixing(h, 0.05f, random) : null;
    }

    /**
     * Starts a spec with plain causal-off softmax attention and no positions.
     *
     * @param heads the number of heads
     * @return the spec
     */
    public static Spec of(int heads) {
        return new Spec(heads, false, 0f, Weighting.SOFTMAX, false, false, 0, false, false, 0);
    }

    /**
     * An immutable layer description; every option returns a new spec. The width is read
     * from the input {@code [steps, features]}.
     */
    public static final class Spec implements LayerSpec {

        private final int heads;
        private final boolean causal;
        private final float rotaryBase;
        private final Weighting weighting;
        private final boolean scalable;
        private final boolean sinkLogit;
        private final int registers;
        private final boolean talking;
        private final boolean selective;
        private final int window;

        private Spec(int heads, boolean causal, float rotaryBase, Weighting weighting, boolean scalable,
                boolean sinkLogit, int registers, boolean talking, boolean selective, int window) {
            this.heads = heads;
            this.causal = causal;
            this.rotaryBase = rotaryBase;
            this.weighting = weighting;
            this.scalable = scalable;
            this.sinkLogit = sinkLogit;
            this.registers = registers;
            this.talking = talking;
            this.selective = selective;
            this.window = window;
        }

        /** @param c whether attention is causal @return the new spec */
        public Spec causal(boolean c) {
            return new Spec(heads, c, rotaryBase, weighting, scalable, sinkLogit, registers, talking, selective,
                    window);
        }

        /** @param base the RoPE base, 0 for none @return the new spec */
        public Spec rotary(float base) {
            return new Spec(heads, causal, base, weighting, scalable, sinkLogit, registers, talking, selective,
                    window);
        }

        /** @param w how scores become weights @return the new spec */
        public Spec weighting(Weighting w) {
            return new Spec(heads, causal, rotaryBase, w, scalable, sinkLogit, registers, talking, selective,
                    window);
        }

        /** @param on whether to use Scalable-Softmax @return the new spec */
        public Spec scalableSoftmax(boolean on) {
            return new Spec(heads, causal, rotaryBase, weighting, on, sinkLogit, registers, talking, selective,
                    window);
        }

        /** @param on whether to add a learned sink logit per head @return the new spec */
        public Spec sinkLogit(boolean on) {
            return new Spec(heads, causal, rotaryBase, weighting, scalable, on, registers, talking, selective,
                    window);
        }

        /** @param n learned key/value register slots @return the new spec */
        public Spec registers(int n) {
            return new Spec(heads, causal, rotaryBase, weighting, scalable, sinkLogit, n, talking, selective,
                    window);
        }

        /** @param on whether to use talking heads @return the new spec */
        public Spec talkingHeads(boolean on) {
            return new Spec(heads, causal, rotaryBase, weighting, scalable, sinkLogit, registers, on, selective,
                    window);
        }

        /** @param on whether to use selective attention (causal only) @return the new spec */
        public Spec selective(boolean on) {
            return new Spec(heads, causal, rotaryBase, weighting, scalable, sinkLogit, registers, talking, on,
                    window);
        }

        /** @param w the sliding window, 0 for full attention @return the new spec */
        public Spec window(int w) {
            return new Spec(heads, causal, rotaryBase, weighting, scalable, sinkLogit, registers, talking, selective,
                    w);
        }

        /**
         * The settings for a width.
         *
         * @param features the model width
         * @return the settings
         */
        public Settings settings(int features) {
            return new Settings(features, heads, causal, rotaryBase, weighting, scalable, sinkLogit, registers,
                    talking, selective, window);
        }

        @Override
        public Shape outputShape(Shape input) {
            return input;
        }

        @Override
        public RefinedAttention create(Shape input, SplittableRandom random) {
            return new RefinedAttention(settings(input.dim(input.rank() - 1)), random);
        }
    }

    @Override
    public Var forward(Var input) {
        MixerOps.requireSequence(input, s.features(), "RefinedAttention");
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        int h = s.heads();
        int r = s.registers();
        Device device = input.value().device();
        Var q = MixerOps.split(input.matmul(wq), h);
        Var k = MixerOps.split(input.matmul(wk), h);
        Var v = MixerOps.split(input.matmul(wv), h);
        if (s.rotaryBase() > 0) {
            q = Functional.rotary(q, s.rotaryBase(), 0);
            k = Functional.rotary(k, s.rotaryBase(), 0);
        }
        float scale = (float) (1 / Math.sqrt(headSize));
        Var scores = q.matmul(k.transpose()).mul(scale);                       // [b, h, t, t]
        if (s.selective()) {
            Var f = SelectiveAttention.penalty(scores.slice(1, 0, 1));
            lastPenalty = f.value();
            scores = scores.sub(f);
        }
        if (r > 0) {
            Var rk = registerKeys.expand(Shape.of(b, h, r, headSize));
            scores = Var.concat(-1, q.matmul(rk.transpose()).mul(scale), scores);
            v = Var.concat(2, registerValues.expand(Shape.of(b, h, r, headSize)), v);
        }
        Tensor keep = RefinementOps.keep(t, t, r, 0, s.causal(), s.window(), device);
        if (restriction != null) {
            Tensor rt = restriction;
            if (r > 0) {
                int[] dims = rt.shape().toArray();
                dims[dims.length - 1] = r;
                rt = Tensor.concat(-1, Tensor.full(Shape.of(dims), 1f, device), rt);
            }
            keep = keep.mul(rt);
        }
        if (s.scalableSoftmax()) {
            scores = ScalableSoftmax.scale(scores, ssmaxGain,
                    ScalableSoftmax.logCounts(t, r + t, r, s.causal(), s.window(), device));
        }
        if (preMix != null) {
            scores = TalkingHeadsAttention.mix(scores, preMix);
        }
        int keys = r + t;
        if (sink != null) {
            scores = Var.concat(-1, scores, sink.reshape(1, h, 1, 1).expand(Shape.of(b, h, t, 1)));
            int[] dims = keep.shape().toArray();
            dims[dims.length - 1] = 1;
            keep = Tensor.concat(-1, keep, Tensor.full(Shape.of(dims), 1f, device));
        }
        Var w = switch (s.weighting()) {
            case SOFTMAX -> Activation.SOFTMAX.forward(scores.add(RefinementOps.additive(keep)));
            case SIGMOID -> SigmoidAttention.weights(scores, keep, keys);
            case SOFTPICK -> Softpick.apply(scores, keep);
            case STICK_BREAKING -> StickBreakingAttention.weights(scores, keep);
        };
        if (sink != null) {
            w = w.slice(-1, 0, keys);
        }
        if (postMix != null) {
            w = TalkingHeadsAttention.mix(w, postMix);
        }
        lastWeights = w.value();
        return MixerOps.merge(w.matmul(v)).matmul(wo);
    }

    /**
     * Limits which keys each query may see, on top of causality and the window, for example to
     * evaluate a cache eviction policy; register slots stay visible.
     *
     * @param keep a 0/1 mask {@code [steps, steps]} or {@code [batch, 1, steps, steps]}, or
     *     {@code null} to lift the limit
     */
    public void restrict(Tensor keep) {
        this.restriction = keep;
    }

    /**
     * Returns the attention weights of the last forward pass.
     *
     * @return {@code [batch, heads, steps, registers + steps]}, or {@code null} before any pass
     */
    public Tensor lastWeights() {
        return lastWeights;
    }

    /**
     * Returns the selective-attention penalty {@code F} of the last forward pass.
     *
     * @return {@code [batch, 1, steps, steps]}, or {@code null} without selective attention
     */
    public Tensor lastPenalty() {
        return lastPenalty;
    }

    /** @return the settings */
    public Settings settings() {
        return s;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, wo));
        for (Var x : new Var[] {ssmaxGain, sink, registerKeys, registerValues, preMix, postMix}) {
            if (x != null) {
                p.add(x);
            }
        }
        return p;
    }

    @Override
    public String toString() {
        return "RefinedAttention(" + s.features() + ", " + s.heads() + " heads, " + s.weighting()
                + (s.rotaryBase() > 0 ? ", rope" : ", nope") + (s.scalableSoftmax() ? ", ssmax" : "")
                + (s.sinkLogit() ? ", sink" : "") + (s.registers() > 0 ? ", registers=" + s.registers() : "")
                + (s.talkingHeads() ? ", talking-heads" : "") + (s.selective() ? ", selective" : "")
                + (s.window() > 0 ? ", window=" + s.window() : "") + ")";
    }
}
