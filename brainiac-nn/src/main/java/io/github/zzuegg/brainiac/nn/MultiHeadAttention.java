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
    private final java.util.EnumMap<Projection, Adapter> adapters = new java.util.EnumMap<>(Projection.class);

    /** The four projections: queries, keys, values, and the output that joins the heads. */
    public enum Projection {
        /** inputs to queries */
        QUERY,
        /** memory to keys */
        KEY,
        /** memory to values */
        VALUE,
        /** joined heads to the output */
        OUTPUT
    }

    /**
     * Something added to a projection's output, such as a LoRA update: {@code x W + b + delta(x)}.
     */
    public interface Adapter {

        /**
         * @param x the projection's input, {@code [..., in]}
         * @return what to add to its output, {@code [..., out]}
         */
        Var delta(Var x);

        /** @return the adapter's trainable parameters */
        List<Var> parameters();
    }

    /**
     * A low-rank adapter (Hu et al., 2021): {@code delta(x) = (alpha / rank) x A B}, with
     * {@code A} small and random and {@code B} zero, so the layer starts unchanged.
     *
     * @param in the projection's input width
     * @param out the projection's output width
     * @param rank the rank of the update
     * @param alpha the scale, often equal to the rank
     * @param random the random source for {@code A}
     * @return the adapter
     */
    public static Adapter lora(int in, int out, int rank, float alpha, SplittableRandom random) {
        Var a = Var.param(Tensor.uniform(Shape.of(in, rank), (float) -Math.sqrt(1.0 / in), (float) Math.sqrt(1.0 / in), random));
        Var b = Var.param(Tensor.zeros(Shape.of(rank, out)));
        float scale = alpha / rank;
        return new Adapter() {
            @Override
            public Var delta(Var x) {
                return x.matmul(a).matmul(b).mul(scale);
            }

            @Override
            public List<Var> parameters() {
                return List.of(a, b);
            }
        };
    }

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
        Adapter adapter = adapters.get(w == wq ? Projection.QUERY : w == wk ? Projection.KEY : Projection.VALUE);
        if (adapter != null) {
            y = y.add(adapter.delta(x));
        }
        return y.reshape(n, t, heads, size).permute(0, 2, 1, 3);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        return forward(inputs, null, training);
    }

    /**
     * Attention with an extra additive bias on the scores, on top of the causal mask, ALiBi
     * and padding mask the settings ask for: a custom mask (such as block-causal attention,
     * {@code -inf} or a large negative number where a query must not look) or learned
     * relative-position biases.
     *
     * @param inputs as for {@link #forward(List, boolean)}
     * @param extra added to the {@code [batch, heads, queries, keys]} scores, broadcasting
     *     (so {@code [queries, keys]} applies to every head), or {@code null}
     * @param training whether dropout is on
     * @return the output
     */
    public Var forward(List<Var> inputs, Tensor extra, boolean training) {
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
        float scale = s.scale() != 0 ? s.scale() : (float) (1 / Math.sqrt(s.headSize()));
        if (extra == null && !s.alibi() && !s.keyMask() && (!training || s.dropout() == 0)) {
            return output(Functional.attention(q, k, v, s.causal(), scale), n, tq);
        }
        Device device = query.value().device();
        Tensor bias = extra;
        if (s.causal()) {
            Tensor c = Functional.causalMask(tq, tk, device);
            bias = bias == null ? c : bias.add(c);
        }
        if (s.alibi()) {
            Tensor a = Functional.alibi(s.heads(), tq, tk, device);
            bias = bias == null ? a : a.add(bias);
        }
        if (s.keyMask()) {
            Tensor m = Functional.keyMask(inputs.get(data).value().reshape(n, tk));
            bias = bias == null ? m : m.add(bias);
        }
        return output(Functional.attention(q, k, v, bias, scale, training ? s.dropout() : 0, random), n, tq);
    }

    /** Joins the heads of {@code out} {@code [batch, heads, steps, dv]} and projects them. */
    private Var output(Var out, int n, int tq) {
        return joined(out.permute(0, 2, 1, 3).reshape(n, tq, settings.heads() * settings.valueSize()));
    }

    /**
     * The keys and values of the steps self-attention has seen so far, so that each new
     * step attends to them without computing them again. Made by {@link KvCache}. Keys are
     * kept before their rotary rotation, which is applied by position within the cache, so
     * a cache that drops steps from the middle still has contiguous positions.
     *
     * <p>Code outside the module can make one to run {@link MultiHeadAttention#step} or
     * {@link TransformerBlock#step} itself, such as a recurrent or retrieval experiment.
     */
    public static final class Cache {

        private final int window;
        private final KvCache.Strategy strategy;
        private final io.github.zzuegg.brainiac.tensor.DType storage;
        private Tensor keys;
        private Tensor values;
        /** For heavy hitters: the attention each cached step has received so far. */
        private float[] received = new float[0];

        /**
         * Creates an empty cache.
         *
         * @param window the most steps it holds
         * @param strategy which steps it drops when full
         * @param storage the type keys and values are kept in
         */
        public Cache(int window, KvCache.Strategy strategy, io.github.zzuegg.brainiac.tensor.DType storage) {
            if (window < 1) {
                throw new IllegalArgumentException("window must be positive: " + window);
            }
            this.window = window;
            this.strategy = strategy;
            this.storage = storage;
        }

        /** Forgets every step. */
        public void reset() {
            keys = null;
            values = null;
            received = new float[0];
        }

        /** @return the steps held */
        public int size() {
            return keys == null ? 0 : keys.shape().dim(2);
        }

        /** The cached steps to keep when {@code total} steps are held and {@code fresh} are new. */
        int[] keep(int total, int fresh, Tensor held) {
            int drop = total - window;
            int[] kept = new int[window];
            switch (strategy) {
                case KvCache.Strategy.Policy p -> {
                    int[] chosen = p.keep(total, fresh, window,
                            p.tracksAttention() ? java.util.Arrays.copyOf(received, total) : new float[0], held);
                    if (chosen.length != window) {
                        throw new IllegalStateException(p + " kept " + chosen.length + " steps of a window of " + window);
                    }
                    for (int i = 0; i < window; i++) {
                        if (chosen[i] < 0 || chosen[i] >= total || i > 0 && chosen[i] <= chosen[i - 1]) {
                            throw new IllegalStateException(p + " must keep increasing steps below " + total);
                        }
                    }
                    for (int i = total - fresh; i < total; i++) {
                        if (java.util.Arrays.binarySearch(chosen, i) < 0) {
                            throw new IllegalStateException(p + " dropped new step " + i);
                        }
                    }
                    System.arraycopy(chosen, 0, kept, 0, window);
                }
                case KvCache.Strategy.Sinks s -> {
                    int sinks = Math.min(s.count(), window - fresh);
                    for (int i = 0; i < sinks; i++) {
                        kept[i] = i;
                    }
                    for (int i = sinks; i < window; i++) {
                        kept[i] = i + drop;
                    }
                }
                case KvCache.Strategy.HeavyHitters h -> {
                    // the most recent steps stay; among the older ones, those that received
                    // the least attention go
                    int recent = Math.max(fresh, Math.min(window - 1, h.recent()));
                    int older = total - recent;
                    Integer[] order = new Integer[older];
                    for (int i = 0; i < older; i++) {
                        order[i] = i;
                    }
                    float[] r = received;
                    // least attention first; ties drop the older step
                    java.util.Arrays.sort(order, (x, y) -> Float.compare(r[x], r[y]));
                    boolean[] gone = new boolean[total];
                    for (int i = 0; i < drop; i++) {
                        gone[order[i]] = true;
                    }
                    int n = 0;
                    for (int i = 0; i < total; i++) {
                        if (!gone[i]) {
                            kept[n++] = i;
                        }
                    }
                }
                default -> {
                    for (int i = 0; i < window; i++) {
                        kept[i] = i + drop;
                    }
                }
            }
            return kept;
        }
    }

    /**
     * Self-attention for new steps only, given the cached earlier ones: with all steps
     * still in the cache, the result for the new steps equals the last rows of
     * {@link #forward} over the whole sequence. No gradients.
     *
     * @param input the new steps, {@code [batch, steps, features]}
     * @param cache the earlier steps; updated
     * @return the output for the new steps
     */
    public Var step(Var input, Cache cache) {
        Settings s = settings;
        if (s.keyMask() || s.queryFeatures() != s.kvFeatures()) {
            throw new IllegalStateException(this + ": only self-attention without a key mask can be cached");
        }
        int n = input.shape().dim(0);
        int tq = input.shape().dim(1);
        io.github.zzuegg.brainiac.tensor.DType type = input.value().dtype();
        Var q = project(input, wq, bq, s.heads(), s.headSize());
        Var k = project(input, wk, bk, s.kvHeads(), s.headSize());
        Var v = project(input, wv, bv, s.kvHeads(), s.valueSize());
        float[] received = cache.received;
        if (cache.keys != null) {
            k = Var.concat(2, Var.of(cache.keys.to(type)), k);
            v = Var.concat(2, Var.of(cache.values.to(type)), v);
            received = java.util.Arrays.copyOf(received, received.length + tq);
        } else {
            received = new float[tq];
        }
        int tk = k.shape().dim(2);
        if (tk > cache.window) {
            int[] kept = cache.keep(tk, tq, k.value());
            float[] ids = new float[kept.length];
            float[] r = new float[kept.length];
            for (int i = 0; i < kept.length; i++) {
                ids[i] = kept[i];
                r[i] = received[kept[i]];
            }
            Tensor index = Tensor.of(Shape.of(kept.length), ids, input.value().device());
            k = k.gather(2, index);
            v = v.gather(2, index);
            received = r;
            tk = kept.length;
        }
        Var qr = q;
        Var kr = k;
        if (s.rotaryBase() > 0) {
            qr = Functional.rotary(q, s.rotaryBase(), tk - tq);
            kr = Functional.rotary(k, s.rotaryBase(), 0);
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
        boolean track = cache.strategy instanceof KvCache.Strategy.HeavyHitters
                || cache.strategy instanceof KvCache.Strategy.Policy p && p.tracksAttention();
        Tensor[] weights = track ? new Tensor[1] : null;
        Var out = track || s.alibi()
                ? Functional.attention(qr, kr, v, bias, scale, 0, null, weights)
                : Functional.attention(qr, kr, v, s.causal() && tq > 1, scale);
        if (track) {
            // the attention each key received, summed over batch, heads and queries
            float[] w = weights[0].sum(0, false).sum(0, false).sum(0, false).toArray();
            for (int i = 0; i < tk; i++) {
                received[i] += w[i];
            }
        }
        cache.keys = cache.storage != null ? k.value().to(cache.storage) : k.value();
        cache.values = cache.storage != null ? v.value().to(cache.storage) : v.value();
        cache.received = received;
        return joined(out.permute(0, 2, 1, 3).reshape(n, tq, s.heads() * s.valueSize()));
    }

    private Var joined(Var heads) {
        Var y = heads.matmul(wo);
        if (bo != null) {
            y = y.add(bo);
        }
        Adapter adapter = adapters.get(Projection.OUTPUT);
        return adapter == null ? y : y.add(adapter.delta(heads));
    }

    /**
     * @param p a projection
     * @return its weight, {@code [in, out]}
     */
    public Var weight(Projection p) {
        return switch (p) {
            case QUERY -> wq;
            case KEY -> wk;
            case VALUE -> wv;
            case OUTPUT -> wo;
        };
    }

    /**
     * @param p a projection
     * @return its bias, or null without biases
     */
    public Var bias(Projection p) {
        return switch (p) {
            case QUERY -> bq;
            case KEY -> bk;
            case VALUE -> bv;
            case OUTPUT -> bo;
        };
    }

    /**
     * Adds an adapter to a projection, replacing any it had; its parameters join
     * {@link #parameters()}. Freeze the base weights (or leave them out of the optimizer) to
     * train only the adapter.
     *
     * @param p the projection
     * @param adapter the adapter, or null to remove it
     */
    public void adapt(Projection p, Adapter adapter) {
        if (adapter == null) {
            adapters.remove(p);
        } else {
            adapters.put(p, adapter);
        }
    }

    /**
     * Adds a LoRA adapter of the right widths to a projection.
     *
     * @param p the projection
     * @param rank the rank
     * @param alpha the scale
     * @param random the random source
     * @return the adapter
     */
    public Adapter lora(Projection p, int rank, float alpha, SplittableRandom random) {
        Var w = weight(p);
        Adapter a = lora(w.shape().dim(0), w.shape().dim(1), rank, alpha, random);
        adapt(p, a);
        return a;
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
        for (Adapter a : adapters.values()) {
            p.addAll(a.parameters());
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
