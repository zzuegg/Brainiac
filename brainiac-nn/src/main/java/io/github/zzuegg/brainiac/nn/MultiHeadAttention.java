package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Multi-head attention over sequences {@code [steps, features]}. With one input it is
 * self-attention; with two ({@code Node.merge(spec, memory)}) the first input gives the
 * queries and the second the keys and values (cross-attention). With
 * {@link Spec#keyMask(boolean) keyMask}, the last input is a {@code [keys]} mask with 1 for
 * real keys and 0 for padding.
 *
 * <pre>{@code
 * MultiHeadAttention.of(8)              // 8 heads, head size = features / 8
 *     .kvHeads(2)                       // grouped-query attention (1 = multi-query)
 *     .headSize(32).valueSize(32)       // per-head query/key and value sizes
 *     .outputSize(256)                  // default: the query features
 *     .causal(true)                     // each step sees only itself and earlier steps
 *     .rotary(10000)                    // RoPE on queries and keys
 *     .alibi(true)                      // or ALiBi distance biases
 *     .dropout(0.1f)                    // on the attention weights
 *     .scale(0.125f)                    // default 1 / sqrt(headSize)
 *     .bias(false)
 *     .keyMask(true);
 * }</pre>
 */
public final class MultiHeadAttention implements MultiInputLayer {

    /**
     * Every setting, as stored in model files.
     *
     * @param queryFeatures the features of the query input
     * @param kvFeatures the features of the key/value input
     * @param heads the query heads
     * @param kvHeads the key/value heads
     * @param headSize the query/key size per head
     * @param valueSize the value size per head
     * @param outputSize the output features
     * @param bias whether projections have biases
     * @param causal whether to mask future steps
     * @param rotaryBase the RoPE base, or 0 for none
     * @param alibi whether to add ALiBi biases
     * @param dropout the attention-weight dropout rate
     * @param scale the score scale, or 0 for {@code 1 / sqrt(headSize)}
     * @param keyMask whether the last input is a key padding mask
     */
    public record Settings(int queryFeatures, int kvFeatures, int heads, int kvHeads, int headSize, int valueSize,
            int outputSize, boolean bias, boolean causal, float rotaryBase, boolean alibi, float dropout, float scale,
            boolean keyMask) {
    }

    private final Settings settings;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var bq;
    private final Var bk;
    private final Var bv;
    private final Var bo;
    private final RandomGenerator random;

    /**
     * Creates the layer with fresh weights.
     *
     * @param s the settings
     * @param init the weight initializer
     * @param random the random source for weights and dropout
     */
    public MultiHeadAttention(Settings s, Init init, SplittableRandom random) {
        if (s.heads() < 1 || s.kvHeads() < 1 || s.heads() % s.kvHeads() != 0) {
            throw new IllegalArgumentException("heads (" + s.heads() + ") must be a positive multiple of kvHeads ("
                    + s.kvHeads() + ")");
        }
        if (s.rotaryBase() > 0 && s.headSize() % 2 != 0) {
            throw new IllegalArgumentException("rotary embeddings need an even head size, got " + s.headSize());
        }
        this.settings = s;
        int qk = s.heads() * s.headSize();
        int kk = s.kvHeads() * s.headSize();
        int vv = s.kvHeads() * s.valueSize();
        int ov = s.heads() * s.valueSize();
        wq = Var.param(init.create(Shape.of(s.queryFeatures(), qk), s.queryFeatures(), qk, random));
        wk = Var.param(init.create(Shape.of(s.kvFeatures(), kk), s.kvFeatures(), kk, random));
        wv = Var.param(init.create(Shape.of(s.kvFeatures(), vv), s.kvFeatures(), vv, random));
        wo = Var.param(init.create(Shape.of(ov, s.outputSize()), ov, s.outputSize(), random));
        bq = s.bias() ? Var.param(Tensor.zeros(Shape.of(qk))) : null;
        bk = s.bias() ? Var.param(Tensor.zeros(Shape.of(kk))) : null;
        bv = s.bias() ? Var.param(Tensor.zeros(Shape.of(vv))) : null;
        bo = s.bias() ? Var.param(Tensor.zeros(Shape.of(s.outputSize()))) : null;
        this.random = random.split();
    }

    /**
     * Describes multi-head attention.
     *
     * @param heads the number of heads
     * @return the spec
     */
    public static Spec of(int heads) {
        return new Spec(heads);
    }

    /** An immutable description; every setter returns a new spec. */
    public static final class Spec implements LayerSpec, MultiInputLayer.Spec {

        private final int heads;
        private int kvHeads = -1;
        private int headSize = -1;
        private int valueSize = -1;
        private int outputSize = -1;
        private boolean bias = true;
        private boolean causal;
        private float rotaryBase;
        private boolean alibi;
        private float dropout;
        private float scale;
        private boolean keyMask;
        private Init init = Init.XAVIER;

        private Spec(int heads) {
            if (heads < 1) {
                throw new IllegalArgumentException("heads must be at least 1, got " + heads);
            }
            this.heads = heads;
        }

        private Spec copy() {
            Spec s = new Spec(heads);
            s.kvHeads = kvHeads;
            s.headSize = headSize;
            s.valueSize = valueSize;
            s.outputSize = outputSize;
            s.bias = bias;
            s.causal = causal;
            s.rotaryBase = rotaryBase;
            s.alibi = alibi;
            s.dropout = dropout;
            s.scale = scale;
            s.keyMask = keyMask;
            s.init = init;
            return s;
        }

        /**
         * Sets the number of key/value heads: fewer than {@code heads} gives grouped-query
         * attention, 1 gives multi-query attention.
         *
         * @param kvHeads a divisor of the head count
         * @return the new spec
         */
        public Spec kvHeads(int kvHeads) {
            Spec s = copy();
            s.kvHeads = kvHeads;
            return s;
        }

        /**
         * Sets the query/key size per head; default: query features / heads.
         *
         * @param size the head size
         * @return the new spec
         */
        public Spec headSize(int size) {
            Spec s = copy();
            s.headSize = size;
            return s;
        }

        /**
         * Sets the value size per head; default: the head size.
         *
         * @param size the value size
         * @return the new spec
         */
        public Spec valueSize(int size) {
            Spec s = copy();
            s.valueSize = size;
            return s;
        }

        /**
         * Sets the output features; default: the query features.
         *
         * @param size the output size
         * @return the new spec
         */
        public Spec outputSize(int size) {
            Spec s = copy();
            s.outputSize = size;
            return s;
        }

        /**
         * Turns projection biases on or off.
         *
         * @param bias whether to learn biases
         * @return the new spec
         */
        public Spec bias(boolean bias) {
            Spec s = copy();
            s.bias = bias;
            return s;
        }

        /**
         * Masks future steps, for autoregressive models.
         *
         * @param causal whether to mask
         * @return the new spec
         */
        public Spec causal(boolean causal) {
            Spec s = copy();
            s.causal = causal;
            return s;
        }

        /**
         * Applies rotary position embeddings to queries and keys.
         *
         * @param base the frequency base, usually 10000; 0 turns it off
         * @return the new spec
         */
        public Spec rotary(float base) {
            Spec s = copy();
            s.rotaryBase = base;
            return s;
        }

        /**
         * Adds ALiBi distance biases.
         *
         * @param alibi whether to add them
         * @return the new spec
         */
        public Spec alibi(boolean alibi) {
            Spec s = copy();
            s.alibi = alibi;
            return s;
        }

        /**
         * Drops attention weights while training.
         *
         * @param rate the share dropped, in [0, 1)
         * @return the new spec
         */
        public Spec dropout(float rate) {
            Dropout.check(rate);
            Spec s = copy();
            s.dropout = rate;
            return s;
        }

        /**
         * Sets the score scale; default {@code 1 / sqrt(headSize)}.
         *
         * @param scale the scale
         * @return the new spec
         */
        public Spec scale(float scale) {
            Spec s = copy();
            s.scale = scale;
            return s;
        }

        /**
         * Expects a key padding mask as the last input.
         *
         * @param mask whether the last input is a mask
         * @return the new spec
         */
        public Spec keyMask(boolean mask) {
            Spec s = copy();
            s.keyMask = mask;
            return s;
        }

        /**
         * Sets the weight initializer.
         *
         * @param init the initializer
         * @return the new spec
         */
        public Spec init(Init init) {
            Spec s = copy();
            s.init = java.util.Objects.requireNonNull(init, "init");
            return s;
        }

        Settings settings(List<Shape> inputs) {
            int data = inputs.size() - (keyMask ? 1 : 0);
            if (data < 1 || data > 2) {
                throw new IllegalArgumentException("attention takes a query input, an optional key/value input and"
                        + (keyMask ? " a mask" : " (with keyMask) a mask") + ", got " + inputs.size() + " inputs");
            }
            Shape q = inputs.get(0);
            Shape kv = inputs.get(data - 1);
            if (q.rank() != 2 || kv.rank() != 2) {
                throw new IllegalArgumentException("attention expects [steps, features] inputs, got " + inputs);
            }
            if (keyMask && inputs.get(data).size() != kv.dim(0)) {
                throw new IllegalArgumentException("the key mask needs one value per key step, got "
                        + inputs.get(data));
            }
            int d = q.dim(1);
            int hs = headSize > 0 ? headSize : d / heads;
            if (hs < 1 || (headSize <= 0 && d % heads != 0)) {
                throw new IllegalArgumentException(heads + " heads do not divide " + d + " features; set headSize");
            }
            return new Settings(d, kv.dim(1), heads, kvHeads > 0 ? kvHeads : heads, hs,
                    valueSize > 0 ? valueSize : hs, outputSize > 0 ? outputSize : d, bias, causal, rotaryBase, alibi,
                    dropout, scale, keyMask);
        }

        @Override
        public Shape outputShape(List<Shape> inputs) {
            Settings s = settings(inputs);
            return Shape.of(inputs.get(0).dim(0), s.outputSize());
        }

        @Override
        public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
            return new MultiHeadAttention(settings(inputs), init, random);
        }

        @Override
        public Shape outputShape(Shape input) {
            if (keyMask) {
                throw new IllegalStateException("with keyMask, wire attention with Node.merge(spec, mask)");
            }
            return outputShape(List.of(input));
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            return create(List.of(input), random);
        }
    }

    private Var project(Var x, Var w, Var b, int heads, int size) {
        int n = x.shape().dim(0);
        int t = x.shape().dim(1);
        Var y = x.matmul(w);
        if (b != null) {
            y = y.add(b);
        }
        return y.reshape(n, t, heads, size).permute(0, 2, 1, 3);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Settings s = settings;
        int data = inputs.size() - (s.keyMask() ? 1 : 0);
        Var query = inputs.get(0);
        Var memory = inputs.get(data - 1);
        if (query.shape().rank() != 3 || query.shape().dim(2) != s.queryFeatures()
                || memory.shape().dim(2) != s.kvFeatures()) {
            throw new IllegalArgumentException(this + " expects queries [batch, steps, " + s.queryFeatures()
                    + "] and keys [batch, steps, " + s.kvFeatures() + "], got " + query.shape() + " and "
                    + memory.shape());
        }
        int n = query.shape().dim(0);
        int tq = query.shape().dim(1);
        int tk = memory.shape().dim(1);
        Var q = project(query, wq, bq, s.heads(), s.headSize());
        Var k = project(memory, wk, bk, s.kvHeads(), s.headSize());
        Var v = project(memory, wv, bv, s.kvHeads(), s.valueSize());
        if (s.rotaryBase() > 0) {
            q = Functional.rotary(q, s.rotaryBase(), tk - tq);
            k = Functional.rotary(k, s.rotaryBase(), 0);
        }
        Device device = query.value().device();
        Tensor bias = null;
        if (s.causal()) {
            bias = Functional.causalMask(tq, tk, device);
        }
        if (s.alibi()) {
            Tensor a = Functional.alibi(s.heads(), tq, tk, device);
            bias = bias == null ? a : a.add(bias);
        }
        if (s.keyMask()) {
            Tensor m = Functional.keyMask(inputs.get(data).value().reshape(n, tk));
            bias = bias == null ? m : m.add(bias);
        }
        float scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(s.headSize()));
        Var out = Functional.attention(q, k, v, bias, scale, training ? s.dropout() : 0, random);
        Var y = out.permute(0, 2, 1, 3).reshape(n, tq, s.heads() * s.valueSize()).matmul(wo);
        return bo != null ? y.add(bo) : y;
    }

    /**
     * The keys and values of the steps self-attention has seen so far, so that each new
     * step attends to them without computing them again. Made by {@link KvCache}.
     */
    static final class Cache {

        private final int window;
        private Tensor keys;
        private Tensor values;
        private int position;

        /** @param window the most steps kept; older ones are dropped */
        Cache(int window) {
            this.window = window;
        }

        void reset() {
            keys = null;
            values = null;
            position = 0;
        }
    }

    /**
     * Self-attention for new steps only, given the cached earlier ones: the result for the
     * new steps equals the last rows of {@link #forward} over the whole sequence (with
     * rotary positions, also once the window slides). No gradients.
     *
     * @param input the new steps, {@code [batch, steps, features]}
     * @param cache the earlier steps; updated
     * @return the output for the new steps
     */
    Var step(Var input, Cache cache) {
        Settings s = settings;
        if (s.keyMask() || s.queryFeatures() != s.kvFeatures()) {
            throw new IllegalStateException(this + ": only self-attention without a key mask can be cached");
        }
        int n = input.shape().dim(0);
        int tq = input.shape().dim(1);
        Var q = project(input, wq, bq, s.heads(), s.headSize());
        Var k = project(input, wk, bk, s.kvHeads(), s.headSize());
        Var v = project(input, wv, bv, s.kvHeads(), s.valueSize());
        if (s.rotaryBase() > 0) {
            q = Functional.rotary(q, s.rotaryBase(), cache.position);
            k = Functional.rotary(k, s.rotaryBase(), cache.position);
        }
        if (cache.keys != null) {
            k = Var.concat(2, Var.of(cache.keys), k);
            v = Var.concat(2, Var.of(cache.values), v);
        }
        int tk = k.shape().dim(2);
        if (tk > cache.window) {
            k = k.slice(2, tk - cache.window, tk);
            v = v.slice(2, tk - cache.window, tk);
            tk = cache.window;
        }
        Device device = input.value().device();
        Tensor bias = null;
        if (s.causal() && tq > 1) {
            bias = Functional.causalMask(tq, tk, device);
        }
        if (s.alibi()) {
            Tensor a = Functional.alibi(s.heads(), tq, tk, device);
            bias = bias == null ? a : a.add(bias);
        }
        float scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(s.headSize()));
        Var out = Functional.attention(q, k, v, bias, scale, 0, null);
        cache.keys = k.value();
        cache.values = v.value();
        cache.position += tq;
        Var y = out.permute(0, 2, 1, 3).reshape(n, tq, s.heads() * s.valueSize()).matmul(wo);
        return bo != null ? y.add(bo) : y;
    }

    @Override
    public Var forward(Var input) {
        return forward(List.of(input), false);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), settings.outputSize());
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), settings.outputSize());
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(wq, wk, wv, wo));
        if (bq != null) {
            p.addAll(List.of(bq, bk, bv, bo));
        }
        return p;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    @Override
    public String toString() {
        return "MultiHeadAttention(" + settings.heads() + " heads" + (settings.kvHeads() != settings.heads()
                ? ", " + settings.kvHeads() + " kv" : "") + (settings.causal() ? ", causal" : "") + ")";
    }

    static Config write(MultiHeadAttention a) {
        Settings s = a.settings;
        return Config.builder()
                .put("queryFeatures", s.queryFeatures()).put("kvFeatures", s.kvFeatures())
                .put("heads", s.heads()).put("kvHeads", s.kvHeads())
                .put("headSize", s.headSize()).put("valueSize", s.valueSize()).put("outputSize", s.outputSize())
                .put("bias", s.bias()).put("causal", s.causal()).put("rotaryBase", s.rotaryBase())
                .put("alibi", s.alibi()).put("dropout", s.dropout()).put("scale", s.scale())
                .put("keyMask", s.keyMask())
                .build();
    }

    static MultiHeadAttention read(Config c) {
        return new MultiHeadAttention(new Settings(c.getInt("queryFeatures"), c.getInt("kvFeatures"),
                c.getInt("heads"), c.getInt("kvHeads"), c.getInt("headSize"), c.getInt("valueSize"),
                c.getInt("outputSize"), c.getBoolean("bias"), c.getBoolean("causal"), c.getFloat("rotaryBase"),
                c.getBoolean("alibi"), c.getFloat("dropout"), c.getFloat("scale"), c.getBoolean("keyMask")),
                Init.ZEROS, new SplittableRandom());
    }
}
