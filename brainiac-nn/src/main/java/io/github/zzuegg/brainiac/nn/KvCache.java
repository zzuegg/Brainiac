package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Generates with a causal transformer one token at a time without recomputing the past.
 * Each attention layer keeps the keys and values of the tokens it has seen, so a new token
 * costs one step through the network instead of a pass over the whole window: for a
 * window of {@code T} tokens, about {@code T} times less work per token.
 *
 * <p>What happens when the window is full is a {@link Strategy}:
 * <ul>
 * <li>{@link Strategy#restart()} (default): start over from the last half window of
 *     tokens, recomputing them as a fresh window. Every token then sees between half and a
 *     whole window of context, exactly as in training, for one short recomputation every
 *     half window.
 * <li>{@link Strategy#sliding()}: drop the oldest token. Cheapest, but a model trained on
 *     windows leans on their first tokens and degrades (StreamingLLM).
 * <li>{@link Strategy#sinks(int)}: keep the first few tokens for good, plus the most recent
 *     (attention sinks, StreamingLLM).
 * <li>{@link Strategy#heavyHitters(int)}: keep the most recent tokens, and of the older ones
 *     those that received the most attention so far (H2O).
 * </ul>
 * Rotary positions count within the cache, so a cache that dropped tokens still has
 * consecutive positions. Keys and values can be stored in a narrower type, such as
 * {@code DType.BFLOAT16}, for half the memory. The model must
 * be a chain of position-wise layers ({@link Embedding}, {@link Dense}, {@link LayerNorm},
 * {@link Dropout}) and causal {@link TransformerBlock}s.
 *
 * <pre>{@code
 * KvCache cache = KvCache.of(model);
 * Tensor p = cache.append(prompt);      // probabilities of the token after the prompt
 * p = cache.append(nextToken);           // one more token, one cheap step
 * }</pre>
 */
public final class KvCache {

    private final Model model;
    private final List<Layer> layers;
    private final List<MultiHeadAttention.Cache> caches = new ArrayList<>();
    private final int window;
    private final Strategy strategy;
    private int[][] recent;
    /** The streams the cache holds, fixed by the first append after a reset; 0 when empty. */
    private int streams;
    private int length;

    /** What a full cache drops; see the class description. */
    public sealed interface Strategy {

        /** @return start over from the last half window (the default) */
        static Strategy restart() {
            return new Restart();
        }

        /** @return drop the oldest token */
        static Strategy sliding() {
            return new Sliding();
        }

        /**
         * @param count the first tokens kept for good, usually 4
         * @return keep the first tokens plus the most recent
         */
        static Strategy sinks(int count) {
            if (count < 1) {
                throw new IllegalArgumentException("keep at least one sink token, got " + count);
            }
            return new Sinks(count);
        }

        /**
         * @param recent the most recent tokens always kept, for example half the window
         * @return keep recent tokens and the older ones that received the most attention
         */
        static Strategy heavyHitters(int recent) {
            if (recent < 1) {
                throw new IllegalArgumentException("keep at least one recent token, got " + recent);
            }
            return new HeavyHitters(recent);
        }

        /**
         * Reads a strategy from its name: {@code restart}, {@code sliding}, {@code sinks:4}
         * or {@code h2o:32} (heavy hitters with 32 recent tokens).
         *
         * @param spec the name
         * @return the strategy
         */
        static Strategy parse(String spec) {
            String[] p = spec.trim().toLowerCase(java.util.Locale.ROOT).split(":");
            return switch (p[0]) {
                case "restart" -> restart();
                case "sliding" -> sliding();
                case "sinks" -> sinks(p.length > 1 ? Integer.parseInt(p[1]) : 4);
                case "h2o", "heavy" -> {
                    if (p.length < 2) {
                        throw new IllegalArgumentException("h2o needs the recent tokens it keeps, like h2o:32");
                    }
                    yield heavyHitters(Integer.parseInt(p[1]));
                }
                default -> throw new IllegalArgumentException("unknown cache strategy " + spec
                        + "; known: restart, sliding, sinks:N, h2o:N");
            };
        }

        /** Start over from the last half window. */
        record Restart() implements Strategy {
            @Override
            public String toString() {
                return "restart";
            }
        }

        /** Drop the oldest token. */
        record Sliding() implements Strategy {
            @Override
            public String toString() {
                return "sliding";
            }
        }

        /** @param count the first tokens kept for good */
        record Sinks(int count) implements Strategy {
            @Override
            public String toString() {
                return "sinks:" + count;
            }
        }

        /** @param recent the most recent tokens always kept */
        record HeavyHitters(int recent) implements Strategy {
            @Override
            public String toString() {
                return "h2o:" + recent;
            }
        }
    }

    private KvCache(Model model, int window, Strategy strategy, io.github.zzuegg.brainiac.tensor.DType storage) {
        if (!model.isChain()) {
            throw new IllegalArgumentException("a KV cache needs a sequential model");
        }
        this.model = model;
        this.layers = model.layers();
        this.window = window;
        this.strategy = java.util.Objects.requireNonNull(strategy, "strategy");
        boolean attention = false;
        for (Layer layer : layers) {
            if (layer instanceof TransformerBlock block) {
                if (!block.settings().causal() || block.settings().memoryFeatures() > 0 || block.settings().keyMask()) {
                    throw new IllegalArgumentException(block + " is not a plain causal block, so it cannot be cached");
                }
                caches.add(new MultiHeadAttention.Cache(window, strategy, storage));
                attention = true;
            } else if (layer instanceof MultiHeadAttention mha) {
                if (!mha.settings().causal() || mha.settings().keyMask()) {
                    throw new IllegalArgumentException(mha + " is not causal self-attention, so it cannot be cached");
                }
                caches.add(new MultiHeadAttention.Cache(window, strategy, storage));
                attention = true;
            } else if (layer instanceof Embedding || layer instanceof Dense || layer instanceof LayerNorm
                    || layer instanceof Dropout) {
                caches.add(null);
            } else {
                throw new IllegalArgumentException(layer + " may mix steps, so a KV cache cannot run it");
            }
        }
        if (!attention) {
            throw new IllegalArgumentException("the model has no causal attention to cache");
        }
    }

    /**
     * A cache with the model's own window, the steps of its input, that restarts from the
     * last half window when full.
     *
     * @param model a sequential causal transformer, {@code [steps]} token ids in
     * @return an empty cache
     */
    public static KvCache of(Model model) {
        return of(model, Strategy.restart());
    }

    /**
     * A cache with the model's own window.
     *
     * @param model a sequential causal transformer
     * @param strategy what a full cache drops
     * @return an empty cache
     */
    public static KvCache of(Model model, Strategy strategy) {
        return of(model, model.inputShape().dim(0), strategy, null);
    }

    /**
     * A cache with every setting.
     *
     * @param model a sequential causal transformer
     * @param window the most tokens each new token attends to
     * @param strategy what a full cache drops
     * @param storage the type keys and values are kept in, or {@code null} for the model's
     * @return an empty cache
     */
    public static KvCache of(Model model, int window, Strategy strategy, io.github.zzuegg.brainiac.tensor.DType storage) {
        if (window < 2) {
            throw new IllegalArgumentException("the window must be at least 2, got " + window);
        }
        return new KvCache(model, window, strategy, storage);
    }

    /**
     * Feeds tokens and returns the model's output for the last of them.
     *
     * @param ids one or more token ids
     * @return the output for the last token, such as next-token probabilities {@code [vocabulary]}
     */
    public Tensor append(int... ids) {
        Tensor out = append(new int[][] {ids});
        return out.reshape(out.shape().dim(1));
    }

    /**
     * Feeds the same number of tokens to each of several streams at once, such as several
     * texts generated together: one pass of the model serves all of them, which costs about
     * as much as one stream on a GPU. The number of streams is fixed until {@link #reset}.
     * The streams share their cache's eviction (all hold the same positions).
     *
     * @param ids per stream, one or more token ids; the same count for every stream
     * @return the output for the last token of each stream, {@code [streams, vocabulary]}
     */
    public Tensor append(int[][] ids) {
        if (ids.length == 0 || ids[0].length == 0) {
            throw new IllegalArgumentException("append needs at least one stream and one token");
        }
        int count = ids[0].length;
        for (int[] row : ids) {
            if (row.length != count) {
                throw new IllegalArgumentException("every stream needs the same number of tokens");
            }
        }
        if (streams != 0 && ids.length != streams) {
            throw new IllegalArgumentException("the cache holds " + streams + " streams, got " + ids.length
                    + "; reset() it first");
        }
        if (streams == 0) {
            streams = ids.length;
            recent = new int[streams][0];
        }
        length += count;
        if (!(strategy instanceof Strategy.Restart)) {
            // a block while it fits, then one at a time, each dropping what the strategy drops
            int room = Math.min(count, Math.max(1, window - size()));
            Tensor out = run(columns(ids, 0, room));
            for (int i = room; i < count; i++) {
                out = run(columns(ids, i, i + 1));
            }
            return out;
        }
        int total = recent[0].length + count;
        int[][] feed = ids;
        int from = 0;
        if (total > window) {
            // start over from the last half window, plus the new tokens
            resetLayers();
            from = total - Math.min(window, window / 2 + count);
        }
        int[][] all = new int[streams][];
        for (int b = 0; b < streams; b++) {
            int[] r = new int[total];
            System.arraycopy(recent[b], 0, r, 0, recent[b].length);
            System.arraycopy(ids[b], 0, r, recent[b].length, count);
            all[b] = java.util.Arrays.copyOfRange(r, from, total);
        }
        if (from > 0) {
            feed = all;
        }
        recent = all;
        return run(feed);
    }

    private static int[][] columns(int[][] ids, int from, int to) {
        int[][] out = new int[ids.length][];
        for (int b = 0; b < ids.length; b++) {
            out[b] = java.util.Arrays.copyOfRange(ids[b], from, to);
        }
        return out;
    }

    /** @return how many tokens the attention layers hold now, at most the window */
    public int size() {
        for (MultiHeadAttention.Cache c : caches) {
            if (c != null) {
                return c.size();
            }
        }
        return 0;
    }

    /** @return what a full cache drops */
    public Strategy strategy() {
        return strategy;
    }

    private Tensor run(int[][] ids) {
        int steps = ids[0].length;
        float[] values = new float[ids.length * steps];
        for (int b = 0; b < ids.length; b++) {
            for (int i = 0; i < steps; i++) {
                values[b * steps + i] = ids[b][i];
            }
        }
        Var x = Var.of(Tensor.of(Shape.of(ids.length, steps), values, model.device())).to(model.dtype());
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            MultiHeadAttention.Cache cache = caches.get(i);
            if (layer instanceof TransformerBlock block) {
                x = block.step(x, cache);
            } else if (layer instanceof MultiHeadAttention mha) {
                x = mha.step(x, cache);
            } else {
                x = layer.forward(x, false);
            }
        }
        int[] dims = x.shape().toArray();
        return x.value().slice(1, steps - 1, steps).reshape(ids.length, dims[dims.length - 1]);
    }

    /** @return how many tokens have been fed since the cache was made or reset */
    public int length() {
        return length;
    }

    /** Forgets every token, to start a new text. */
    public void reset() {
        resetLayers();
        recent = null;
        streams = 0;
        length = 0;
    }

    private void resetLayers() {
        for (MultiHeadAttention.Cache c : caches) {
            if (c != null) {
                c.reset();
            }
        }
    }
}
