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
 * <p>A model trained on windows has only seen texts that start at the window's first token,
 * and it leans on those first tokens (a plain sliding window of keys falls apart, as
 * StreamingLLM found). So when the window is full, the cache starts over from the last
 * half window of tokens, recomputing them as a fresh window: every token then sees between
 * half and a whole window of context, exactly as in training, at the cost of one short
 * recomputation every half window. The model must
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
    private int[] recent = new int[0];
    private int length;

    private KvCache(Model model, int window) {
        if (!model.isChain()) {
            throw new IllegalArgumentException("a KV cache needs a sequential model");
        }
        this.model = model;
        this.layers = model.layers();
        this.window = window;
        boolean attention = false;
        for (Layer layer : layers) {
            if (layer instanceof TransformerBlock block) {
                if (!block.settings().causal() || block.settings().memoryFeatures() > 0 || block.settings().keyMask()) {
                    throw new IllegalArgumentException(block + " is not a plain causal block, so it cannot be cached");
                }
                caches.add(new MultiHeadAttention.Cache(window));
                attention = true;
            } else if (layer instanceof MultiHeadAttention mha) {
                if (!mha.settings().causal() || mha.settings().keyMask()) {
                    throw new IllegalArgumentException(mha + " is not causal self-attention, so it cannot be cached");
                }
                caches.add(new MultiHeadAttention.Cache(window));
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
     * A cache with the model's own window, the steps of its input.
     *
     * @param model a sequential causal transformer, {@code [steps]} token ids in
     * @return an empty cache
     */
    public static KvCache of(Model model) {
        return new KvCache(model, model.inputShape().dim(0));
    }

    /**
     * A cache that keeps a given number of steps.
     *
     * @param model a sequential causal transformer
     * @param window the most tokens each new token attends to; the model's own window
     *     unless it was trained on longer texts
     * @return an empty cache
     */
    public static KvCache of(Model model, int window) {
        if (window < 1) {
            throw new IllegalArgumentException("the window must be at least 1, got " + window);
        }
        return new KvCache(model, window);
    }

    /**
     * Feeds tokens and returns the model's output for the last of them.
     *
     * @param ids one or more token ids
     * @return the output for the last token, such as next-token probabilities {@code [vocabulary]}
     */
    public Tensor append(int... ids) {
        if (ids.length == 0) {
            throw new IllegalArgumentException("append needs at least one token");
        }
        length += ids.length;
        int[] all = new int[recent.length + ids.length];
        System.arraycopy(recent, 0, all, 0, recent.length);
        System.arraycopy(ids, 0, all, recent.length, ids.length);
        int[] feed = ids;
        if (all.length > window) {
            // start over from the last half window, plus the new tokens
            resetLayers();
            int keep = Math.min(window, window / 2 + ids.length);
            feed = java.util.Arrays.copyOfRange(all, all.length - keep, all.length);
            all = feed;
        }
        recent = all;
        return run(feed);
    }

    private Tensor run(int[] ids) {
        float[] values = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            values[i] = ids[i];
        }
        Var x = Var.of(Tensor.of(Shape.of(1, ids.length), values, model.device())).to(model.dtype());
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
        int steps = dims[1];
        return x.value().slice(1, steps - 1, steps).reshape(dims[dims.length - 1]);
    }

    /** @return how many tokens have been fed since the cache was made or reset */
    public int length() {
        return length;
    }

    /** Forgets every token, to start a new text. */
    public void reset() {
        resetLayers();
        recent = new int[0];
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
